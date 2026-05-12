import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Base64;
import java.util.HashSet;
import java.util.Set;

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
        for (int i = 0; i < args.length; i++)
        {
            switch (args[i])
            {
                case "--lud":   ludPath   = args[++i]; break;
                case "--tasks": tasksPath = args[++i]; break;
                default: System.err.println("unexpected arg: " + args[i]); System.exit(2);
            }
        }
        if (ludPath == null || tasksPath == null)
        {
            System.err.println("usage: RenderTrials --lud <path> --tasks <file>");
            System.exit(2);
        }

        final Game game = GameLoader.loadGameFromFile(new File(ludPath));
        if (game == null) { System.err.println("could not load: " + ludPath); System.exit(1); }
        game.setMaxMoveLimit(10_000);

        final PrintStream out = new PrintStream(System.out, false, "UTF-8");
        final Base64.Encoder b64 = Base64.getEncoder();

        int taskIdx = 0;
        try (BufferedReader br = new BufferedReader(new FileReader(tasksPath)))
        {
            String line;
            while ((line = br.readLine()) != null)
            {
                if (line.isEmpty()) { taskIdx++; continue; }
                final String[] parts = line.split("\t", -1);
                if (parts.length != 3)
                {
                    System.err.println("bad task line " + taskIdx + ": " + line);
                    System.exit(3);
                }
                final byte[] rng = hexToBytes(parts[0]);
                final int[] moves = parseInts(parts[1]);
                final int[] plies = parseInts(parts[2]);
                renderTask(game, rng, moves, plies, taskIdx, out, b64);
                taskIdx++;
            }
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
        out.print("POS ");
        out.print(taskIdx); out.print(' ');
        out.print(ply); out.print(' ');
        out.print(mover); out.print(' ');
        out.print(terminal); out.print(' ');
        out.println(enc);
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
