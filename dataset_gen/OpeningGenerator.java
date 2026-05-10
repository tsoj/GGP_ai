import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Random;

import org.apache.commons.rng.core.RandomProviderDefaultState;

import game.Game;
import main.collections.FastArrayList;
import other.context.Context;
import other.move.Move;
import other.trial.Trial;

/**
 * Produces a list of opening move-index sequences to use as starting
 * positions for self-play, so we don't waste compute playing the same
 * game over and over.
 *
 * Strategy: sample random playouts of growing length, deduplicate by the
 * resulting state hash, and grow the length whenever new samples mostly
 * collide with positions we've already collected.
 *
 * Each returned sequence is an int[] of legal-move indices: the i-th entry
 * is the index into `game.moves(context).moves()` at the i-th opening ply.
 * Replaying the indices against the same start RNG reproduces the position.
 */
public final class OpeningGenerator
{
    private OpeningGenerator() {}

    public static List<int[]> generate(
        final Game game,
        final byte[] startRngState,
        final int count,
        final int maxDepth,
        final long randomSeed)
    {
        if (count <= 1)
        {
            final List<int[]> single = new ArrayList<>(1);
            single.add(new int[0]);
            return single;
        }

        final Random rng = new Random(randomSeed);
        // LinkedHashMap keeps insertion order so the dataset is deterministic.
        final LinkedHashMap<Long, int[]> uniqueByHash = new LinkedHashMap<>();
        // Always include the root position itself.
        uniqueByHash.put(hashAfter(game, startRngState, new int[0]), new int[0]);

        final HashSet<Long> seenAtDepth = new HashSet<>();

        for (int depth = 1; depth <= maxDepth; depth++)
        {
            seenAtDepth.clear();
            // Per-depth budget: enough to give us a good shot at filling the
            // quota, but bounded so deep games don't sample forever.
            final int budget = Math.max(256, count * 8);
            final int sinceNewLimit = Math.max(64, budget / 4);
            int samples = 0;
            int sinceNew = 0;
            int newAtDepth = 0;

            while (samples < budget)
            {
                samples++;
                final int[] seq = sampleSequence(game, startRngState, depth, rng);
                if (seq == null) continue; // game ended before reaching `depth`

                final long h = hashAfter(game, startRngState, seq);
                if (!seenAtDepth.add(h))
                {
                    sinceNew++;
                }
                else if (uniqueByHash.putIfAbsent(h, seq) == null)
                {
                    newAtDepth++;
                    sinceNew = 0;
                    if (uniqueByHash.size() >= count) break;
                }
                else
                {
                    sinceNew++;
                }

                if (sinceNew >= sinceNewLimit)
                    break; // depth has stalled, grow it
            }

            System.out.println("  [openings] depth=" + depth
                + " samples=" + samples
                + " new=" + newAtDepth
                + " total=" + uniqueByHash.size() + "/" + count);

            if (uniqueByHash.size() >= count)
            {
                final List<int[]> out = new ArrayList<>(uniqueByHash.values());
                return out.subList(0, count);
            }
        }

        System.out.println("  [openings] hit maxDepth=" + maxDepth
            + " with " + uniqueByHash.size() + "/" + count + " unique");
        return new ArrayList<>(uniqueByHash.values());
    }

    private static int[] sampleSequence(
        final Game game, final byte[] startRng, final int depth, final Random rng)
    {
        final Context ctx = freshContext(game, startRng);
        final int[] seq = new int[depth];
        for (int i = 0; i < depth; i++)
        {
            if (ctx.trial().over()) return null;
            final FastArrayList<Move> legal = game.moves(ctx).moves();
            if (legal.isEmpty()) return null;
            final int idx = rng.nextInt(legal.size());
            seq[i] = idx;
            game.apply(ctx, legal.get(idx));
        }
        return seq;
    }

    private static long hashAfter(
        final Game game, final byte[] startRng, final int[] seq)
    {
        final Context ctx = freshContext(game, startRng);
        for (int i = 0; i < seq.length; i++)
        {
            if (ctx.trial().over()) return Long.MIN_VALUE;
            final FastArrayList<Move> legal = game.moves(ctx).moves();
            final int idx = seq[i];
            if (idx < 0 || idx >= legal.size()) return Long.MIN_VALUE;
            game.apply(ctx, legal.get(idx));
        }
        return ctx.state().fullHash(ctx);
    }

    private static Context freshContext(final Game game, final byte[] startRngState)
    {
        final Trial trial = new Trial(game);
        final Context ctx = new Context(game, trial);
        ctx.rng().restoreState(new RandomProviderDefaultState(startRngState.clone()));
        game.start(ctx);
        return ctx;
    }
}
