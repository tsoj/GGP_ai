import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import org.apache.commons.rng.core.RandomProviderDefaultState;

import game.Game;
import main.collections.FastArrayList;
import other.GameLoader;
import other.context.Context;
import other.move.Move;
import other.trial.Trial;

/**
 * Replays trials of a single .lud and prints the serialized text form of
 * specified plies. One JVM, many trials.
 *
 * Args:
 *   --lud <path>      Game definition file (required).
 *   --tasks <file>    Each line: <rngHex>\t<movesCSV>\t<pliesCSV>
 *
 * For every requested ply in every task, writes one line to stdout:
 *   POS <taskIdx> <ply> <mover> <terminal> <base64Utf8Text>
 *
 * Ply 0 is the initial position; ply k is the position after the k-th move
 * in MOVES. <terminal> is 1 if trial.over() at that ply, else 0. <mover> is
 * the side-to-move id (0 if no mover, e.g. terminal positions).
 */
public class RenderTrials
{
    public static void main(final String[] args) throws Exception
    {
        String ludPath = null;
        String tasksPath = null;
        int threads = Math.max(1, Runtime.getRuntime().availableProcessors());
        for (int i = 0; i < args.length; i++)
        {
            switch (args[i])
            {
                case "--lud":     ludPath   = args[++i]; break;
                case "--tasks":   tasksPath = args[++i]; break;
                case "--threads": threads   = Math.max(1, Integer.parseInt(args[++i])); break;
                default: System.err.println("unexpected arg: " + args[i]); System.exit(2);
            }
        }
        if (ludPath == null || tasksPath == null)
        {
            System.err.println("usage: RenderTrials --lud <path> --tasks <file> [--threads N]");
            System.exit(2);
        }

        final Game game = GameLoader.loadGameFromFile(new File(ludPath));
        if (game == null) { System.err.println("could not load: " + ludPath); System.exit(1); }
        game.setMaxMoveLimit(10_000);

        // Read all tasks into memory so workers can pick them up by index.
        final List<byte[]> rngs = new ArrayList<>();
        final List<int[]> moves = new ArrayList<>();
        final List<int[]> plies = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(tasksPath)))
        {
            String line;
            int taskIdx = 0;
            while ((line = br.readLine()) != null)
            {
                if (line.isEmpty()) { rngs.add(null); moves.add(null); plies.add(null); taskIdx++; continue; }
                final String[] parts = line.split("\t", -1);
                if (parts.length != 3)
                {
                    System.err.println("bad task line " + taskIdx + ": " + line);
                    System.exit(3);
                }
                rngs.add(hexToBytes(parts[0]));
                moves.add(parseInts(parts[1]));
                plies.add(parseInts(parts[2]));
                taskIdx++;
            }
        }

        final PrintStream out = new PrintStream(System.out, false, "UTF-8");
        // Single shared base64 encoder is thread-safe (stateless after construction).
        final Base64.Encoder b64 = Base64.getEncoder();

        final int nTasks = rngs.size();
        final ExecutorService pool = Executors.newFixedThreadPool(threads);
        try
        {
            final List<Future<?>> futures = new ArrayList<>(nTasks);
            for (int i = 0; i < nTasks; i++)
            {
                if (rngs.get(i) == null) continue;
                final int idx = i;
                final byte[] rng = rngs.get(i);
                final int[] mv = moves.get(i);
                final int[] pl = plies.get(i);
                futures.add(pool.submit(() -> renderTask(game, rng, mv, pl, idx, out, b64)));
            }
            for (final Future<?> f : futures)
            {
                try { f.get(); }
                catch (final Exception e)
                {
                    System.err.println("task failed: " + e);
                    System.exit(7);
                }
            }
        }
        finally
        {
            pool.shutdown();
            pool.awaitTermination(1, TimeUnit.DAYS);
        }
        out.flush();
    }

    private static void renderTask(
        final Game game, final byte[] rng, final int[] moves, final int[] plies,
        final int taskIdx, final PrintStream out, final Base64.Encoder b64)
    {
        // Sanity: plies must be ascending and within [0, moves.length].
        final Set<Integer> want = new HashSet<>();
        int maxPly = 0;
        for (final int p : plies)
        {
            if (p < 0 || p > moves.length)
            {
                System.err.println("ply out of range: task=" + taskIdx + " ply=" + p
                    + " (moves=" + moves.length + ")");
                System.exit(4);
            }
            want.add(p);
            if (p > maxPly) maxPly = p;
        }

        final Trial trial = new Trial(game);
        final Context ctx = new Context(game, trial);
        ctx.rng().restoreState(new RandomProviderDefaultState(rng.clone()));
        game.start(ctx);

        if (want.contains(0)) emitPosition(ctx, 0, taskIdx, out, b64);

        for (int k = 0; k < maxPly; k++)
        {
            if (trial.over())
            {
                System.err.println("trial ended early at ply=" + k + " (task=" + taskIdx + ")");
                System.exit(5);
            }
            final FastArrayList<Move> legal = game.moves(ctx).moves();
            final int idx = moves[k];
            if (idx < 0 || idx >= legal.size())
            {
                System.err.println("illegal move index at ply=" + k + " (task=" + taskIdx
                    + ", idx=" + idx + ", legal=" + legal.size() + ")");
                System.exit(6);
            }
            game.apply(ctx, legal.get(idx));
            final int ply = k + 1;
            if (want.contains(ply)) emitPosition(ctx, ply, taskIdx, out, b64);
        }
    }

    private static void emitPosition(
        final Context ctx, final int ply, final int taskIdx,
        final PrintStream out, final Base64.Encoder b64)
    {
        final int mover = ctx.state() != null ? ctx.state().mover() : 0;
        final int terminal = ctx.trial().over() ? 1 : 0;
        final String text = PositionSerializer.serialize(ctx);
        final String enc = b64.encodeToString(text.getBytes(StandardCharsets.UTF_8));
        // Build the whole line then println in one call: PrintStream.println is
        // synchronized, so each line stays atomic across worker threads.
        final StringBuilder sb = new StringBuilder(enc.length() + 32);
        sb.append("POS ").append(taskIdx).append(' ').append(ply).append(' ')
          .append(mover).append(' ').append(terminal).append(' ').append(enc);
        out.println(sb.toString());
    }

    private static int[] parseInts(final String csv)
    {
        if (csv.isEmpty()) return new int[0];
        final String[] parts = csv.split(",");
        final int[] out = new int[parts.length];
        for (int i = 0; i < parts.length; i++) out[i] = Integer.parseInt(parts[i].trim());
        return out;
    }

    private static byte[] hexToBytes(final String s)
    {
        if ((s.length() & 1) != 0) throw new IllegalArgumentException("bad hex: " + s);
        final byte[] b = new byte[s.length() / 2];
        for (int i = 0; i < b.length; i++)
        {
            b[i] = (byte) ((Character.digit(s.charAt(2 * i), 16) << 4)
                          | Character.digit(s.charAt(2 * i + 1), 16));
        }
        return b;
    }
}
