import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.commons.rng.core.RandomProviderDefaultState;

import game.Game;
import main.collections.FastArrayList;
import other.GameLoader;
import other.context.Context;
import other.move.Move;
import other.trial.Trial;

/**
 * Iterates over many .lud files, plays a few random plies in each,
 * runs PositionSerializer, and writes one TSV row per game with the
 * outcome: ok / load_fail / serialize_fail / play_fail. Lets us see
 * which games the serializer can't currently handle.
 *
 * Args:
 *   --manifest <path>     File with one .lud path per line.
 *   --plies N             Random plies before serialization (default: 4).
 *   --seed N              RNG seed (default: 42).
 *   --out <path>          TSV output (default: scan.tsv).
 *   --samples-dir <path>  If set, write one .txt with the rendered
 *                         position per successful game (debug).
 *   --threads N           Worker threads (default: cores).
 */
public class ScanPositions
{
    public static void main(final String[] args) throws Exception
    {
        String manifest = null;
        int plies = 4;
        long seed = 42L;
        String out = "scan.tsv";
        String samplesDir = null;
        int threads = Runtime.getRuntime().availableProcessors();

        for (int i = 0; i < args.length; i++)
        {
            switch (args[i])
            {
                case "--manifest":    manifest   = args[++i]; break;
                case "--plies":       plies      = Integer.parseInt(args[++i]); break;
                case "--seed":        seed       = Long.parseLong(args[++i]); break;
                case "--out":         out        = args[++i]; break;
                case "--samples-dir": samplesDir = args[++i]; break;
                case "--threads":     threads    = Integer.parseInt(args[++i]); break;
                default: System.err.println("unexpected: " + args[i]); System.exit(2);
            }
        }
        if (manifest == null) { System.err.println("--manifest required"); System.exit(2); }

        final List<String> ludPaths = new ArrayList<>();
        try (BufferedReader rdr = new BufferedReader(new FileReader(manifest)))
        {
            String line;
            while ((line = rdr.readLine()) != null)
            {
                line = line.trim();
                if (!line.isEmpty() && !line.startsWith("#")) ludPaths.add(line);
            }
        }

        if (samplesDir != null) new File(samplesDir).mkdirs();
        final File outFile = new File(out);
        if (outFile.getParentFile() != null) outFile.getParentFile().mkdirs();

        final int fPlies = plies;
        final long fSeed = seed;
        final String fSamplesDir = samplesDir;
        final AtomicInteger doneCt = new AtomicInteger();
        final AtomicInteger okCt = new AtomicInteger();
        final Map<String, Integer> failCounts = new HashMap<>();
        final List<String> rows = new ArrayList<>(ludPaths.size());
        for (int i = 0; i < ludPaths.size(); i++) rows.add(null);

        final ExecutorService pool = Executors.newFixedThreadPool(Math.max(1, threads));
        final List<Future<?>> futs = new ArrayList<>();
        final long t0 = System.nanoTime();
        for (int i = 0; i < ludPaths.size(); i++)
        {
            final int idx = i;
            final String ludPath = ludPaths.get(i);
            futs.add(pool.submit(() ->
            {
                final Result r = scanOne(ludPath, fPlies, fSeed + idx, fSamplesDir);
                rows.set(idx, r.toTsv(ludPath));
                if ("ok".equals(r.status)) okCt.incrementAndGet();
                else synchronized (failCounts) { failCounts.merge(r.status, 1, Integer::sum); }
                final int d = doneCt.incrementAndGet();
                if (d % 25 == 0 || d == ludPaths.size())
                {
                    final double el = (System.nanoTime() - t0) / 1e9;
                    System.err.printf("[%d/%d] ok=%d  (%.1fs)%n",
                        d, ludPaths.size(), okCt.get(), el);
                }
            }));
        }
        pool.shutdown();
        for (final Future<?> f : futs) { try { f.get(); } catch (final Exception ignored) {} }
        pool.awaitTermination(1, TimeUnit.DAYS);

        try (PrintWriter w = new PrintWriter(new FileWriter(outFile)))
        {
            w.println("status\tstage\tlud\tdetail");
            for (final String row : rows) if (row != null) w.println(row);
        }

        System.err.println();
        System.err.printf("total=%d  ok=%d  fail=%d%n",
            ludPaths.size(), okCt.get(), ludPaths.size() - okCt.get());
        for (final Map.Entry<String, Integer> e : failCounts.entrySet())
            System.err.printf("  %-22s %d%n", e.getKey(), e.getValue());
        System.err.println("rows -> " + outFile.getAbsolutePath());
    }

    private static Result scanOne(
        final String ludPath, final int plies, final long seed, final String samplesDir)
    {
        final Game game;
        try { game = GameLoader.loadGameFromFile(new File(ludPath)); }
        catch (final Throwable t) { return Result.fail("load_fail", "load", oneLine(t)); }
        if (game == null) return Result.fail("load_fail", "load", "null");
        game.setMaxMoveLimit(10_000);

        final Trial trial = new Trial(game);
        final Context ctx = new Context(game, trial);
        try
        {
            ctx.rng().restoreState(new RandomProviderDefaultState(longToBytes(seed)));
            game.start(ctx);
            final Random rng = new Random(seed);
            for (int p = 0; p < plies; p++)
            {
                if (trial.over()) break;
                final FastArrayList<Move> legal = game.moves(ctx).moves();
                if (legal.isEmpty()) break;
                game.apply(ctx, legal.get(rng.nextInt(legal.size())));
            }
        }
        catch (final Throwable t) { return Result.fail("play_fail", "play", oneLine(t)); }

        final String text;
        try { text = PositionSerializer.serialize(ctx); }
        catch (final Throwable t) { return Result.fail("serialize_fail", "serialize", oneLine(t)); }

        if (samplesDir != null)
        {
            final String name = new File(ludPath).getName().replaceAll("\\.lud$", "") + ".txt";
            try (PrintWriter w = new PrintWriter(new FileWriter(new File(samplesDir, name))))
            { w.print(text); }
            catch (final Exception e)
            { return Result.fail("write_fail", "write", oneLine(e)); }
        }
        return Result.ok();
    }

    private static byte[] longToBytes(final long v)
    {
        final byte[] b = new byte[8];
        for (int i = 0; i < 8; i++) b[i] = (byte) (v >>> (8 * i));
        return b;
    }

    private static String oneLine(final Throwable t)
    {
        final String s = t.getClass().getSimpleName() + ": " + t.getMessage();
        return s.replace('\n', ' ').replace('\t', ' ');
    }
    private static String oneLine(final Object t) { return String.valueOf(t).replace('\n', ' ').replace('\t', ' '); }

    private static final class Result
    {
        final String status, stage, detail;
        Result(final String s, final String st, final String d) { status = s; stage = st; detail = d; }
        static Result ok() { return new Result("ok", "", ""); }
        static Result fail(final String s, final String st, final String d) { return new Result(s, st, d); }
        String toTsv(final String lud)
        {
            return status + '\t' + stage + '\t' + lud + '\t' + (detail == null ? "" : detail);
        }
    }
}
