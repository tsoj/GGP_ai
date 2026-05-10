import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

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
 * Trials are stored in a compact format: a header (game path, RNG seed,
 * opening prefix length, ranking) plus a single line of move indices. The
 * indices are positions in `game.moves(context).moves()` at each ply, so the
 * full game can be replayed deterministically against the same Ludii build.
 *
 * Args:
 *   --out-dir <dir>           Root output dir (one subdir per game).
 *   --num-games <int>         Trials per .lud.
 *   --num-playouts <int>      MCTS iterations per move.
 *   --max-seconds <float>     Wall-clock cap per move (default: no cap).
 *   --move-limit <int>        Max plies per trial (default: 1000).
 *   --opening-max-depth <int> Cap for opening enumeration (default: 8).
 *   --seed <long>             Base RNG seed (default: 42).
 *   --manifest <file>         File with one .lud path per line.
 *   [trailing positional args] Additional .lud paths.
 */
public class GenerateDataset
{
    private static void deleteRecursively(final File f)
    {
        if (f == null || !f.exists()) return;
        if (f.isDirectory())
        {
            final File[] children = f.listFiles();
            if (children != null)
                for (final File c : children) deleteRecursively(c);
        }
        f.delete();
    }

    public static void main(final String[] args) throws Exception
    {
        String outDir = "trials";
        int numGames = 10;
        int numPlayouts = 1000;
        double maxSeconds = -1.0;
        int moveLimit = 1000;
        int openingMaxDepth = 8;
        long baseSeed = 42L;
        String manifestPath = null;
        final List<String> ludPaths = new ArrayList<>();

        for (int i = 0; i < args.length; i++)
        {
            switch (args[i])
            {
                case "--out-dir":           outDir          = args[++i]; break;
                case "--num-games":         numGames        = Integer.parseInt(args[++i]); break;
                case "--num-playouts":      numPlayouts     = Integer.parseInt(args[++i]); break;
                case "--max-seconds":       maxSeconds      = Double.parseDouble(args[++i]); break;
                case "--move-limit":        moveLimit       = Integer.parseInt(args[++i]); break;
                case "--opening-max-depth": openingMaxDepth = Integer.parseInt(args[++i]); break;
                case "--seed":              baseSeed        = Long.parseLong(args[++i]); break;
                case "--manifest":          manifestPath    = args[++i]; break;
                default:                    ludPaths.add(args[i]);
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
        final double thinkSeconds = (maxSeconds > 0) ? maxSeconds : Double.MAX_VALUE;

        for (final String ludPath : ludPaths)
        {
            final File ludFile = new File(ludPath);
            if (!ludFile.exists())
            {
                System.err.println("Skip (missing): " + ludPath);
                continue;
            }

            final String gameId = ludFile.getName().replaceAll("\\.lud$", "");
            final File gameOutDir = new File(outDir, gameId);
            gameOutDir.mkdirs();

            final Game game;
            try
            {
                game = GameLoader.loadGameFromFile(ludFile);
            }
            catch (final Throwable t)
            {
                System.err.println("Failed to load " + ludPath + ": " + t.getMessage());
                continue;
            }
            if (game == null)
            {
                System.err.println("Failed to compile: " + ludPath);
                continue;
            }
            game.setMaxMoveLimit(moveLimit);
            final int numPlayers = game.players().count();

            // Probe UCT support up front so we can fail fast on the whole game.
            {
                final MCTS probe = MCTS.createUCT();
                if (!probe.supportsGame(game))
                {
                    System.out.println("[skip game] UCT does not support " + gameId);
                    deleteRecursively(gameOutDir);
                    continue;
                }
            }

            System.out.println("=== " + gameId + " (" + numPlayers + " players) ===");

            // One RNG state per game so opening enumeration is reproducible.
            final byte[] startRng = freshRngState(game, baseSeed ^ gameId.hashCode());

            final List<int[]> openings;
            try
            {
                openings = OpeningGenerator.generate(
                    game, startRng, numGames, openingMaxDepth, baseSeed);
            }
            catch (final Throwable t)
            {
                System.err.println("[skip game] opening generator crashed: " + t);
                deleteRecursively(gameOutDir);
                continue;
            }

            boolean gameAborted = false;
            for (int g = 0; g < openings.size(); g++)
            {
                final int[] opening = openings.get(g);
                final File trialFile = new File(gameOutDir, "trial_" + g + ".txt");
                if (trialFile.exists())
                {
                    System.out.println("  [skip] trial_" + g + " exists");
                    continue;
                }

                final List<AI> ais = new ArrayList<>();
                ais.add(null);
                for (int p = 1; p <= numPlayers; p++)
                    ais.add(MCTS.createUCT());

                final Trial trial = new Trial(game);
                final Context ctx = new Context(game, trial);
                ctx.rng().restoreState(new RandomProviderDefaultState(startRng.clone()));
                game.start(ctx);

                // Replay opening prefix (no AI involvement).
                final int[] moves = new int[Math.min(opening.length + moveLimit, 1 << 20)];
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
                    System.err.println("  [skip game] opening replay failed at trial " + g);
                    gameAborted = true;
                    break;
                }

                // Init AIs only after the opening is in place (some agents
                // do per-game setup that depends on the start state, but
                // UCT just looks at the game; this is fine).
                for (int p = 1; p <= numPlayers; p++)
                    ais.get(p).initAI(game, p);

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
                                "AI returned a move that isn't in the legal list");
                        }
                        moves[moveCount++] = idx;
                        game.apply(ctx, legal.get(idx));
                    }
                }
                catch (final Throwable t)
                {
                    System.err.println("  [skip game] playout " + g + " crashed: " + t);
                    closeAll(ais);
                    deleteRecursively(gameOutDir);
                    gameAborted = true;
                    break;
                }

                writeTrial(trialFile, ludFile, startRng, opening.length,
                           Arrays.copyOf(moves, moveCount), trial);
                System.out.println("  saved " + trialFile.getName()
                    + "  opening_plies=" + opening.length
                    + "  moves=" + moveCount
                    + "  status=" + (trial.status() == null ? "null" : trial.status().toString()));

                closeAll(ais);
            }

            if (gameAborted) continue;
        }
    }

    private static byte[] freshRngState(final Game game, final long seed)
    {
        final Trial t = new Trial(game);
        final Context c = new Context(game, t);
        // Use a fresh RNG seeded explicitly so the same .lud + seed yields
        // the same start state every time we run the script.
        c.rng().restoreState(new RandomProviderDefaultState(longToBytes(seed)));
        game.start(c);
        return ((RandomProviderDefaultState) c.rng().saveState()).getState().clone();
    }

    private static byte[] longToBytes(final long v)
    {
        // ApacheCommons RNG state is opaque bytes; for the default provider
        // an 8-byte seed is accepted. Use a stable encoding.
        final byte[] b = new byte[8];
        for (int i = 0; i < 8; i++) b[i] = (byte) (v >>> (8 * i));
        return b;
    }

    /** Defensive copy used as input to MCTS so its rollouts don't mutate ctx. */
    private static Context copyContext(final Context ctx)
    {
        return new Context(ctx);
    }

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

    private static void writeTrial(
        final File file, final File ludFile, final byte[] startRng,
        final int openingPlies, final int[] moves, final Trial trial)
    {
        file.getParentFile().mkdirs();
        try (PrintWriter w = new PrintWriter(file))
        {
            w.println("GAME=" + ludFile.getAbsolutePath());
            w.println("LUDII_VERSION=" + Constants.LUDEME_VERSION);
            w.println("RNG=" + bytesToHex(startRng));
            w.println("OPENING_PLIES=" + openingPlies);
            w.println("STATUS=" + (trial.status() == null ? "" : trial.status().toString()));
            final double[] ranking = trial.ranking();
            final StringBuilder rb = new StringBuilder();
            if (ranking != null)
            {
                for (int i = 1; i < ranking.length; i++)
                {
                    if (i > 1) rb.append(',');
                    rb.append(ranking[i]);
                }
            }
            w.println("RANKING=" + rb);
            final StringBuilder mb = new StringBuilder("MOVES=");
            for (int i = 0; i < moves.length; i++)
            {
                if (i > 0) mb.append(',');
                mb.append(moves[i]);
            }
            w.println(mb);
        }
        catch (final Exception e)
        {
            System.err.println("  [error] could not save " + file + ": " + e);
        }
    }

    private static String bytesToHex(final byte[] b)
    {
        final StringBuilder sb = new StringBuilder(b.length * 2);
        for (final byte v : b) sb.append(String.format("%02x", v & 0xff));
        return sb.toString();
    }
}
