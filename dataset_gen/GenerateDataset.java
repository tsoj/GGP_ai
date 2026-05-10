import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.rng.core.RandomProviderDefaultState;

import game.Game;
import main.Constants;
import main.collections.FastArrayList;
import other.AI;
import other.GameLoader;
import other.context.Context;
import other.move.Move;
import other.trial.Trial;
import search.mcts.MCTS;

/**
 * Generate UCT self-play trials for one or more .lud files.
 *
 * Output is a single dataset file (one record per trial) plus a failures
 * log listing every game that was skipped, with a reason. Trials are
 * stored in a compact format: a header (game path, RNG seed, opening
 * prefix length, ranking) plus a single line of move indices, where each
 * index is a position in `game.moves(context).moves()` at that ply, so the
 * full game can be replayed deterministically against the same Ludii build.
 *
 * A game is skipped (and noted in the failures log) when:
 *  - .lud fails to load or compile
 *  - UCT does not support the game
 *  - the opening generator can't produce the requested number of unique
 *    starting positions
 *  - the very first trial takes longer than --max-game-seconds
 *  - after --drawish-check-after games one outcome dominates
 *    (>= --drawish-threshold of trials)
 *  - any playout crashes
 *
 * Args:
 *   --out-dir <dir>              Dataset root. Each kept game gets its own
 *                                subdirectory containing trials.txt with
 *                                all trials for that game.
 *   --failures-file <path>       Skipped-games log (default: failures.tsv).
 *   --num-games <int>            Trials per .lud (default: 50).
 *   --num-playouts <int>         MCTS iterations per move (default: 1000).
 *   --max-seconds <float>        Wall-clock cap per move (default: no cap).
 *   --move-limit <int>           Max plies per trial; longer = draw (default: 500).
 *   --max-game-seconds <float>   Skip game if first trial exceeds this (default: 0.5).
 *   --drawish-check-after <int>  Run drawish check after this many trials (default: 50).
 *   --drawish-threshold <float>  Skip game if dominant outcome >= this (default: 0.9).
 *   --opening-max-depth <int>    Cap for opening enumeration (default: 8).
 *   --seed <long>                Base RNG seed (default: 42).
 *   --manifest <file>            File with one .lud path per line.
 *   [trailing positional args]   Additional .lud paths.
 */
public class GenerateDataset
{
    public static void main(final String[] args) throws Exception
    {
        String outDir = "dataset";
        String failuresFile = "failures.tsv";
        int numGames = 50;
        int numPlayouts = 1000;
        double maxSeconds = -1.0;
        int moveLimit = 500;
        double maxGameSeconds = 0.5;
        int drawishCheckAfter = 50;
        double drawishThreshold = 0.9;
        int openingMaxDepth = 8;
        long baseSeed = 42L;
        String manifestPath = null;
        final List<String> ludPaths = new ArrayList<>();

        for (int i = 0; i < args.length; i++)
        {
            switch (args[i])
            {
                case "--out-dir":              outDir            = args[++i]; break;
                case "--failures-file":        failuresFile      = args[++i]; break;
                case "--num-games":            numGames          = Integer.parseInt(args[++i]); break;
                case "--num-playouts":         numPlayouts       = Integer.parseInt(args[++i]); break;
                case "--max-seconds":          maxSeconds        = Double.parseDouble(args[++i]); break;
                case "--move-limit":           moveLimit         = Integer.parseInt(args[++i]); break;
                case "--max-game-seconds":     maxGameSeconds    = Double.parseDouble(args[++i]); break;
                case "--drawish-check-after":  drawishCheckAfter = Integer.parseInt(args[++i]); break;
                case "--drawish-threshold":    drawishThreshold  = Double.parseDouble(args[++i]); break;
                case "--opening-max-depth":    openingMaxDepth   = Integer.parseInt(args[++i]); break;
                case "--seed":                 baseSeed          = Long.parseLong(args[++i]); break;
                case "--manifest":             manifestPath      = args[++i]; break;
                default:                       ludPaths.add(args[i]);
            }
        }

        if (manifestPath != null)
        {
            try (BufferedReader rdr = new BufferedReader(new FileReader(manifestPath)))
            {
                String line;
                while ((line = rdr.readLine()) != null)
                {
                    line = line.trim();
                    if (!line.isEmpty() && !line.startsWith("#"))
                        ludPaths.add(line);
                }
            }
        }

        if (ludPaths.isEmpty())
        {
            System.err.println("No .lud paths supplied (use --manifest or positional args).");
            System.exit(2);
        }

        new File(outDir).mkdirs();
        ensureParent(new File(failuresFile));

        final double thinkSeconds = (maxSeconds > 0) ? maxSeconds : Double.MAX_VALUE;

        // Append to the failures log so re-runs can extend it; the Python
        // wrapper truncates it up front when it wants a clean run.
        try (PrintWriter failuresOut = new PrintWriter(new FileWriter(failuresFile, true)))
        {
            for (final String ludPath : ludPaths)
                processGame(ludPath, outDir, failuresOut,
                    numGames, numPlayouts, thinkSeconds, moveLimit,
                    maxGameSeconds, drawishCheckAfter, drawishThreshold,
                    openingMaxDepth, baseSeed);
        }
    }

    private static void processGame(
        final String ludPath,
        final String outDir,
        final PrintWriter failuresOut,
        final int numGames,
        final int numPlayouts,
        final double thinkSeconds,
        final int moveLimit,
        final double maxGameSeconds,
        final int drawishCheckAfter,
        final double drawishThreshold,
        final int openingMaxDepth,
        final long baseSeed)
    {
        final File ludFile = new File(ludPath);
        if (!ludFile.exists())
        {
            logFailure(failuresOut, ludPath, "missing", "");
            return;
        }
        final String gameId = ludFile.getName().replaceAll("\\.lud$", "");

        final Game game;
        try
        {
            game = GameLoader.loadGameFromFile(ludFile);
        }
        catch (final Throwable t)
        {
            logFailure(failuresOut, ludPath, "load_error", oneLine(t.toString()));
            return;
        }
        if (game == null)
        {
            logFailure(failuresOut, ludPath, "compile_error", "");
            return;
        }
        game.setMaxMoveLimit(moveLimit);
        final int numPlayers = game.players().count();

        if (!MCTS.createUCT().supportsGame(game))
        {
            logFailure(failuresOut, ludPath, "uct_unsupported", "");
            System.out.println("[skip] " + gameId + ": UCT unsupported");
            return;
        }

        System.out.println("=== " + gameId + " (" + numPlayers + " players) ===");
        final byte[] startRng = freshRngState(game, baseSeed ^ gameId.hashCode());

        final List<int[]> openings;
        try
        {
            openings = OpeningGenerator.generate(
                game, startRng, numGames, openingMaxDepth, baseSeed);
        }
        catch (final Throwable t)
        {
            logFailure(failuresOut, ludPath, "opening_error", oneLine(t.toString()));
            return;
        }
        if (openings.size() < numGames)
        {
            logFailure(failuresOut, ludPath, "insufficient_openings",
                "got=" + openings.size() + " want=" + numGames);
            System.out.println("[skip] " + gameId + ": only "
                + openings.size() + "/" + numGames + " unique openings");
            return;
        }

        // Buffer trials in memory so we can drop the whole game atomically
        // if it's later flagged as too slow / too drawish / crashing.
        final List<String> pendingRecords = new ArrayList<>(numGames);
        final Map<String, Integer> outcomeCounts = new HashMap<>();

        for (int g = 0; g < openings.size(); g++)
        {
            final int[] opening = openings.get(g);
            final List<AI> ais = new ArrayList<>();
            ais.add(null);
            for (int p = 1; p <= numPlayers; p++)
                ais.add(MCTS.createUCT());

            final Trial trial = new Trial(game);
            final Context ctx = new Context(game, trial);
            ctx.rng().restoreState(new RandomProviderDefaultState(startRng.clone()));
            game.start(ctx);

            // Replay the opening prefix without AI.
            final int[] moves = new int[opening.length + moveLimit + 8];
            int moveCount = 0;
            boolean openingOk = true;
            for (final int idx : opening)
            {
                if (trial.over()) { openingOk = false; break; }
                final FastArrayList<Move> legal = game.moves(ctx).moves();
                if (idx < 0 || idx >= legal.size()) { openingOk = false; break; }
                moves[moveCount++] = idx;
                game.apply(ctx, legal.get(idx));
            }
            if (!openingOk)
            {
                closeAll(ais);
                logFailure(failuresOut, ludPath, "opening_replay_failed",
                    "trial=" + g);
                return;
            }

            for (int p = 1; p <= numPlayers; p++)
                ais.get(p).initAI(game, p);

            final long t0 = System.nanoTime();
            try
            {
                while (!trial.over())
                {
                    final int mover = ctx.state().mover();
                    final FastArrayList<Move> legal = game.moves(ctx).moves();
                    final Move chosen = ais.get(mover).selectAction(
                        game, copyContext(ctx), thinkSeconds, numPlayouts, -1);
                    final int idx = indexOf(legal, chosen);
                    if (idx < 0)
                    {
                        throw new IllegalStateException(
                            "AI returned a move not in the legal list");
                    }
                    moves[moveCount++] = idx;
                    game.apply(ctx, legal.get(idx));
                }
            }
            catch (final Throwable t)
            {
                closeAll(ais);
                logFailure(failuresOut, ludPath, "playout_crash",
                    "trial=" + g + " err=" + oneLine(t.toString()));
                return;
            }
            final double elapsed = (System.nanoTime() - t0) / 1e9;
            closeAll(ais);

            // First-trial wall-clock cutoff.
            if (g == 0 && elapsed > maxGameSeconds)
            {
                logFailure(failuresOut, ludPath, "too_slow",
                    String.format("first_trial_seconds=%.3f limit=%.3f",
                        elapsed, maxGameSeconds));
                System.out.println("[skip] " + gameId
                    + String.format(": first trial took %.3fs (> %.3f)",
                        elapsed, maxGameSeconds));
                return;
            }

            final String statusStr = trial.status() == null
                ? "" : trial.status().toString();
            outcomeCounts.merge(statusStr, 1, Integer::sum);

            pendingRecords.add(buildRecord(
                ludFile, startRng, opening.length,
                Arrays.copyOf(moves, moveCount), trial, statusStr));

            System.out.println(String.format(
                "  trial_%d  opening_plies=%d  moves=%d  status=%s  (%.3fs)",
                g, opening.length, moveCount, statusStr, elapsed));

            // Drawish check after the configured number of trials.
            final int played = g + 1;
            if (played == drawishCheckAfter)
            {
                int dominant = 0;
                String dominantKey = "";
                for (final Map.Entry<String, Integer> e : outcomeCounts.entrySet())
                {
                    if (e.getValue() > dominant)
                    {
                        dominant = e.getValue();
                        dominantKey = e.getKey();
                    }
                }
                final double frac = (double) dominant / played;
                if (frac >= drawishThreshold)
                {
                    logFailure(failuresOut, ludPath, "too_drawish",
                        String.format("dominant=%.3f outcome=%s after=%d",
                            frac, oneLine(dominantKey), played));
                    System.out.println("[skip] " + gameId
                        + String.format(": %.0f%% same outcome after %d trials",
                            frac * 100, played));
                    return;
                }
            }
        }

        // All trials passed: commit to <outDir>/<gameId>/trials.txt.
        final File gameDir = new File(outDir, gameId);
        gameDir.mkdirs();
        final File trialsFile = new File(gameDir, "trials.txt");
        try (PrintWriter w = new PrintWriter(new FileWriter(trialsFile)))
        {
            w.println("PATH=" + ludFile.getAbsolutePath());
            w.println("LUDII_VERSION=" + Constants.LUDEME_VERSION);
            w.println("NUM_PLAYERS=" + numPlayers);
            w.println("NUM_TRIALS=" + pendingRecords.size());
            for (final String rec : pendingRecords) w.print(rec);
        }
        catch (final Exception e)
        {
            logFailure(failuresOut, ludPath, "write_error", oneLine(e.toString()));
            System.err.println("[skip] " + gameId + ": write failed: " + e);
            return;
        }
        System.out.println("[ok]   " + gameId + ": "
            + pendingRecords.size() + " trials -> " + trialsFile);
    }

    // --------------------------------------------------------- record I/O

    private static String buildRecord(
        final File ludFile, final byte[] startRng,
        final int openingPlies, final int[] moves,
        final Trial trial, final String statusStr)
    {
        final StringBuilder sb = new StringBuilder(8 * moves.length + 256);
        sb.append("---TRIAL---\n");
        sb.append("RNG=").append(bytesToHex(startRng)).append('\n');
        sb.append("OPENING_PLIES=").append(openingPlies).append('\n');
        sb.append("STATUS=").append(statusStr).append('\n');
        sb.append("RANKING=");
        final double[] ranking = trial.ranking();
        if (ranking != null)
        {
            for (int i = 1; i < ranking.length; i++)
            {
                if (i > 1) sb.append(',');
                sb.append(ranking[i]);
            }
        }
        sb.append('\n');
        sb.append("MOVES=");
        for (int i = 0; i < moves.length; i++)
        {
            if (i > 0) sb.append(',');
            sb.append(moves[i]);
        }
        sb.append('\n');
        return sb.toString();
    }

    private static synchronized void logFailure(
        final PrintWriter w, final String ludPath,
        final String reason, final String detail)
    {
        w.print(ludPath);
        w.print('\t');
        w.print(reason);
        w.print('\t');
        w.println(detail);
        w.flush();
    }

    // ----------------------------------------------------------- helpers

    private static byte[] freshRngState(final Game game, final long seed)
    {
        final Trial t = new Trial(game);
        final Context c = new Context(game, t);
        c.rng().restoreState(new RandomProviderDefaultState(longToBytes(seed)));
        game.start(c);
        return ((RandomProviderDefaultState) c.rng().saveState()).getState().clone();
    }

    private static byte[] longToBytes(final long v)
    {
        final byte[] b = new byte[8];
        for (int i = 0; i < 8; i++) b[i] = (byte) (v >>> (8 * i));
        return b;
    }

    private static Context copyContext(final Context ctx) { return new Context(ctx); }

    private static int indexOf(final FastArrayList<Move> moves, final Move m)
    {
        for (int i = 0; i < moves.size(); i++)
            if (moves.get(i).equals(m)) return i;
        return -1;
    }

    private static void closeAll(final List<AI> ais)
    {
        for (final AI ai : ais) if (ai != null) ai.closeAI();
    }

    private static void ensureParent(final File f)
    {
        final File parent = f.getAbsoluteFile().getParentFile();
        if (parent != null) parent.mkdirs();
    }

    private static String bytesToHex(final byte[] b)
    {
        final StringBuilder sb = new StringBuilder(b.length * 2);
        for (final byte v : b) sb.append(String.format("%02x", v & 0xff));
        return sb.toString();
    }

    private static String oneLine(final String s)
    {
        return s == null ? "" : s.replace('\n', ' ').replace('\t', ' ');
    }
}
