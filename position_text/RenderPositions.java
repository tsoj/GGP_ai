import java.io.File;
import java.util.Random;

import org.apache.commons.rng.core.RandomProviderDefaultState;

import game.Game;
import main.collections.FastArrayList;
import other.GameLoader;
import other.context.Context;
import other.move.Move;
import other.trial.Trial;

/**
 * CLI driver that loads a .lud, optionally plays a few random plies to
 * reach a non-trivial position, and prints the serialized text form.
 *
 * Args:
 *   <ludPath>          Path to a .lud file (required).
 *   --plies N          Random plies to play before serializing (default: 0).
 *   --seed N           RNG seed (default: 42).
 *   --num N            Print this many independent positions (default: 1).
 */
public class RenderPositions
{
    public static void main(final String[] args) throws Exception
    {
        String ludPath = null;
        int plies = 0;
        long seed = 42L;
        int num = 1;

        for (int i = 0; i < args.length; i++)
        {
            switch (args[i])
            {
                case "--plies": plies = Integer.parseInt(args[++i]); break;
                case "--seed":  seed  = Long.parseLong(args[++i]); break;
                case "--num":   num   = Integer.parseInt(args[++i]); break;
                default:
                    if (ludPath == null) ludPath = args[i];
                    else { System.err.println("unexpected: " + args[i]); System.exit(2); }
            }
        }
        if (ludPath == null)
        {
            System.err.println("usage: RenderPositions <ludPath> [--plies N] [--seed N] [--num N]");
            System.exit(2);
        }

        final Game game = GameLoader.loadGameFromFile(new File(ludPath));
        if (game == null) { System.err.println("could not load: " + ludPath); System.exit(1); }
        game.setMaxMoveLimit(10_000);

        for (int g = 0; g < num; g++)
        {
            final Trial trial = new Trial(game);
            final Context ctx = new Context(game, trial);
            ctx.rng().restoreState(new RandomProviderDefaultState(longToBytes(seed ^ (g * 0x9E3779B97F4A7C15L))));
            game.start(ctx);

            final Random rng = new Random(seed + g);
            for (int p = 0; p < plies; p++)
            {
                if (trial.over()) break;
                final FastArrayList<Move> legal = game.moves(ctx).moves();
                if (legal.isEmpty()) break;
                game.apply(ctx, legal.get(rng.nextInt(legal.size())));
            }

            if (num > 1)
                System.out.println("###### position " + g + " (plies=" + plies + ", seed=" + (seed + g) + ") ######");
            System.out.println(PositionSerializer.serialize(ctx));
        }
    }

    private static byte[] longToBytes(final long v)
    {
        final byte[] b = new byte[8];
        for (int i = 0; i < 8; i++) b[i] = (byte) (v >>> (8 * i));
        return b;
    }
}
