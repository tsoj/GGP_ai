import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.rng.core.RandomProviderDefaultState;

import game.Game;
import other.AI;
import other.GameLoader;
import other.context.Context;
import other.model.Model;
import other.trial.Trial;
import search.mcts.MCTS;

/**
 * Generate UCT self-play trials for one or more .lud files.
 *
 * Args:
 *   --out-dir <dir>           Root output directory (one subdir per game).
 *   --num-games <int>         Number of games to play per .lud file.
 *   --num-playouts <int>      MCTS iterations per move (UCT playouts).
 *   --max-seconds <float>     Optional hard wall-clock cap per move (default: no cap).
 *   --move-limit <int>        Max moves per trial before forced draw (default: 1000).
 *   --manifest <file>         File containing one .lud path per line.
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
        String manifestPath = null;
        final List<String> ludPaths = new ArrayList<>();

        for (int i = 0; i < args.length; i++)
        {
            switch (args[i])
            {
                case "--out-dir":      outDir       = args[++i]; break;
                case "--num-games":    numGames     = Integer.parseInt(args[++i]); break;
                case "--num-playouts": numPlayouts  = Integer.parseInt(args[++i]); break;
                case "--max-seconds":  maxSeconds   = Double.parseDouble(args[++i]); break;
                case "--move-limit":   moveLimit    = Integer.parseInt(args[++i]); break;
                case "--manifest":     manifestPath = args[++i]; break;
                default:               ludPaths.add(args[i]);
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

            System.out.println("=== " + gameId + " (" + numPlayers + " players) ===");

            for (int g = 0; g < numGames; g++)
            {
                final File trialFile = new File(gameOutDir, "trial_" + g + ".txt");
                if (trialFile.exists())
                {
                    System.out.println("  [skip] trial_" + g + " exists");
                    continue;
                }

                final List<AI> ais = new ArrayList<>();
                ais.add(null);
                boolean uctOk = true;
                for (int p = 1; p <= numPlayers; p++)
                {
                    final MCTS uct = MCTS.createUCT();
                    if (!uct.supportsGame(game))
                    {
                        uctOk = false;
                        break;
                    }
                    ais.add(uct);
                }
                if (!uctOk)
                {
                    System.out.println("  [skip game] UCT does not support " + gameId);
                    // Remove partially-created game directory so it stays out of the dataset.
                    final File[] children = gameOutDir.listFiles();
                    if (children == null || children.length == 0)
                        gameOutDir.delete();
                    break;
                }

                final Trial trial = new Trial(game);
                final Context context = new Context(game, trial);
                final byte[] startRNG =
                        ((RandomProviderDefaultState) context.rng().saveState()).getState();
                game.start(context);

                for (int p = 1; p <= numPlayers; p++)
                    ais.get(p).initAI(game, p);

                final Model model = context.model();
                try
                {
                    while (!trial.over())
                        model.startNewStep(context, ais, thinkSeconds, numPlayouts, -1, 0.0);
                }
                catch (final Throwable t)
                {
                    System.err.println("  [skip game] playout " + g + " crashed: " + t);
                    for (final AI ai : ais)
                        if (ai != null) ai.closeAI();
                    deleteRecursively(gameOutDir);
                    break;
                }

                try
                {
                    trial.saveTrialToTextFile(
                        trialFile,
                        ludFile.getAbsolutePath(),
                        new ArrayList<String>(),
                        new RandomProviderDefaultState(startRNG));
                    System.out.println("  saved " + trialFile.getName()
                        + "  moves=" + trial.numMoves()
                        + "  status=" + (trial.status() == null ? "null" : trial.status().toString()));
                }
                catch (final Exception e)
                {
                    System.err.println("  [error] could not save " + trialFile + ": " + e.getMessage());
                }

                for (final AI ai : ais)
                    if (ai != null) ai.closeAI();
            }
        }
    }
}
