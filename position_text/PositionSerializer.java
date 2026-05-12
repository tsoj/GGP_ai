import java.awt.geom.Point2D;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.TreeSet;

import game.Game;
import game.equipment.component.Component;
import game.equipment.container.Container;
import game.types.board.SiteType;
import other.context.Context;
import other.state.State;
import other.state.container.ContainerState;
import other.topology.TopologyElement;

/**
 * Serializes a Ludii Context into text intended as input to an LLM
 * embedding model. The output has three parts:
 *
 *   LEGEND - per-position glyph -> "P<owner> <pieceName>" mapping. When
 *            a player has only one piece type, that player's owner-glyph
 *            is used (O for P1, X for P2, ...). When a player has
 *            several piece types, single-letter glyphs derived from the
 *            piece name are used (uppercase for P1, lowercase for P2).
 *   FACTS  - Ludii-flavored fact list using the same vocabulary the .lud
 *            uses (who / count / stack), one fact per line, sorted by
 *            label. Site labels in non-board containers are prefixed
 *            with the container name so they don't collide.
 *   BOARD  - ASCII rendering of the main board container, laid out from
 *            each site's centroid so it works for square, hex,
 *            triangular, and other Ludii topologies. Stacked or
 *            multi-count sites overlay a depth digit on the glyph.
 *
 * Past move history is not included.
 */
public final class PositionSerializer
{
    // Owner-glyph palette indexed by player id (0 = unowned).
    // Owner glyph by player id (0 = unowned). Sized for up to 16 players;
    // tweak if Ludii ever ships games with more.
    private static final char[] PLAYER_GLYPHS = {
        '.', 'O', 'X', '#', '&', '@', '%', '+', '*',
        '!', '~', '^', '$', '/', '<', '>', '='
    };
    private static final char EMPTY = '.';
    private static final char OFFBOARD = ' ';
    private static final char STACK_PAD = ' ';

    private PositionSerializer() {}

    public static String serialize(final Context ctx)
    {
        final GlyphMap gm = buildGlyphMap(ctx);
        final StringBuilder out = new StringBuilder(4096);
        appendMeta(out, ctx);
        out.append('\n');
        appendLegend(out, gm);
        out.append('\n');
        appendFacts(out, ctx);
        out.append('\n');
        appendBoard(out, ctx, gm);
        return out.toString();
    }

    // ------------------------------------------------------------- meta

    private static void appendMeta(final StringBuilder out, final Context ctx)
    {
        final Game game = ctx.game();
        final State st = ctx.state();
        out.append("=== META ===\n");
        out.append("game: ").append(game.name()).append('\n');
        out.append("players: ").append(game.players().count()).append('\n');
        out.append("mover: P").append(st.mover()).append('\n');
        out.append("turn: ").append(st.numTurn()).append('\n');
        if (st.currentPhase(st.mover()) != 0)
            out.append("phase: ").append(st.currentPhase(st.mover())).append('\n');
        if (st.temp() != Integer.MIN_VALUE && st.temp() != 0)
            out.append("var: ").append(st.temp()).append('\n');
        final Container board = ctx.containers()[0];
        out.append("board: ").append(describeBoard(board)).append('\n');
        out.append("site_type: ").append(board.defaultSite().name()).append('\n');
    }

    private static String describeBoard(final Container board)
    {
        final String name = board.name();
        return name == null ? "Board" : name;
    }

    // ----------------------------------------------------------- legend

    private static void appendLegend(final StringBuilder out, final GlyphMap gm)
    {
        out.append("=== LEGEND ===\n");
        if (gm.legend.isEmpty())
        {
            out.append("(no pieces on board)\n");
            return;
        }
        for (final String line : gm.legend) out.append(line).append('\n');
    }

    // ------------------------------------------------------------ facts

    private static void appendFacts(final StringBuilder out, final Context ctx)
    {
        out.append("=== FACTS ===\n");
        final Game game = ctx.game();
        final Container[] containers = ctx.containers();
        final Component[] components = ctx.components();
        final ContainerState[] cstates = ctx.state().containerStates();

        final List<String> lines = new ArrayList<>();
        for (int ci = 0; ci < containers.length; ci++)
        {
            final Container cont = containers[ci];
            final SiteType siteType = cont.defaultSite();
            final ContainerState cs = cstates[ci];
            final List<? extends TopologyElement> elements = siteElements(game, cont, siteType);
            for (final TopologyElement el : elements)
            {
                final int idx = el.index();
                final int who = cs.who(idx, siteType);
                if (who <= 0) continue;
                final int what = cs.what(idx, siteType);
                final int count = cs.count(idx, siteType);
                final int stackSize = cs.sizeStack(idx, siteType);
                final String label = qualifiedLabel(ci, cont, el);

                if (stackSize > 1)
                {
                    final StringBuilder stk = new StringBuilder("(stack ").append(label).append(" [");
                    for (int lv = 0; lv < stackSize; lv++)
                    {
                        if (lv > 0) stk.append(' ');
                        final int w = cs.what(idx, lv, siteType);
                        final int o = cs.who(idx, lv, siteType);
                        stk.append('P').append(o).append(':').append(pieceName(components, w));
                    }
                    stk.append("])");
                    lines.add(stk.toString());
                }
                else
                {
                    lines.add("(who " + label + " P" + who + " " + pieceName(components, what) + ")");
                    if (count > 1) lines.add("(count " + label + " " + count + ")");
                }
            }
        }
        // Dedupe defensive: identical-fact collisions (shouldn't happen with
        // container prefixes, but cheap safety net).
        final TreeSet<String> uniq = new TreeSet<>(lines);
        for (final String l : uniq) out.append(l).append('\n');
        if (uniq.isEmpty()) out.append("(empty)\n");
    }

    // ------------------------------------------------------------ board

    private static void appendBoard(final StringBuilder out, final Context ctx, final GlyphMap gm)
    {
        out.append("=== BOARD ===\n");
        final Game game = ctx.game();
        final Container board = ctx.containers()[0];
        final SiteType siteType = board.defaultSite();
        final List<? extends TopologyElement> sites = siteElements(game, board, siteType);
        if (sites.isEmpty())
            throw new IllegalStateException("board has no sites for type " + siteType);

        final ContainerState cs = ctx.state().containerStates()[0];

        // Bounding box.
        double xmin = Double.POSITIVE_INFINITY, xmax = Double.NEGATIVE_INFINITY;
        double ymin = Double.POSITIVE_INFINITY, ymax = Double.NEGATIVE_INFINITY;
        for (final TopologyElement el : sites)
        {
            final Point2D c = el.centroid();
            xmin = Math.min(xmin, c.getX()); xmax = Math.max(xmax, c.getX());
            ymin = Math.min(ymin, c.getY()); ymax = Math.max(ymax, c.getY());
        }

        // Row spacing first: median nonzero y-gap across all sites.
        final double uy = axisUnit(sites, false);
        if (uy <= 0) throw new IllegalStateException("could not determine row spacing");
        final double yTol = uy * 0.4;

        // Group sites into y-buckets ordered top-to-bottom (high y first).
        final TreeMap<Long, java.util.List<TopologyElement>> rowBuckets =
            new TreeMap<>(java.util.Comparator.reverseOrder());
        for (final TopologyElement el : sites)
        {
            final long bucket = Math.round((el.centroid().getY() - ymin) / uy);
            rowBuckets.computeIfAbsent(bucket, k -> new java.util.ArrayList<>()).add(el);
        }

        // Column spacing: median of *within-row* x-gaps. This treats the
        // half-step between hex rows as a row offset (rendered as half-cell
        // indent) rather than as a column of its own.
        final double ux = withinRowUx(rowBuckets);
        if (ux <= 0) throw new IllegalStateException("could not determine column spacing");

        // Detect whether any site in the current position has stack depth
        // or count > 1; if not, use 2-char cells (glyph + sep) instead of
        // 3-char (glyph + depth-digit + sep) to keep wide boards compact.
        final boolean hasDepth = anyDepth(ctx);
        final int glyphSlots = hasDepth ? 2 : 1; // chars at cell start
        // Render with two text columns per cell unit so half-cell offsets
        // (hex rows) can shift by one char without overlap.
        final int textColsPerCell = 2 * glyphSlots;

        // Compute per-row indent in text chars from each row's leftmost x.
        // Row width = (cell count - 1) * textColsPerCell + glyphSlots.
        // Sparsity check uses total board sites vs. text grid area.
        int globalWidth = 0;
        final int rows = rowBuckets.size();
        final int[] rowIndent = new int[rows];
        final int[] rowEnd = new int[rows];
        int rIdx = 0;
        for (final java.util.List<TopologyElement> bucket : rowBuckets.values())
        {
            double rowXmin = Double.POSITIVE_INFINITY, rowXmax = Double.NEGATIVE_INFINITY;
            for (final TopologyElement el : bucket)
            {
                final double x = el.centroid().getX();
                if (x < rowXmin) rowXmin = x;
                if (x > rowXmax) rowXmax = x;
            }
            final int indent = (int) Math.round((rowXmin - xmin) / ux * textColsPerCell);
            final int end = (int) Math.round((rowXmax - xmin) / ux * textColsPerCell) + glyphSlots;
            rowIndent[rIdx] = indent;
            rowEnd[rIdx]   = end;
            if (end > globalWidth) globalWidth = end;
            rIdx++;
        }

        if ((long) globalWidth * rows > 200_000)
            throw new IllegalStateException(
                "ASCII grid too large: " + globalWidth + "x" + rows
                + " (sites=" + sites.size() + ", ux=" + ux + ", uy=" + uy + ")");

        // Detect non-grid layouts (spiral, fractal, scattered): when the
        // average row has fewer than two sites, the y-axis isn't really
        // "rows" — it's just a continuous coordinate. The grid would be
        // mostly whitespace and wouldn't communicate spatial structure.
        // Emit a compact occupied-site listing instead.
        final double sitesPerRow = sites.size() / Math.max(1.0, (double) rows);
        if (sitesPerRow < 2.0 && rows >= 8)
        {
            appendCompact(out, ctx, gm, sites, siteType, cs);
            return;
        }

        final char[][] grid = new char[rows][Math.max(1, globalWidth)];
        for (int r = 0; r < rows; r++) Arrays.fill(grid[r], OFFBOARD);

        rIdx = 0;
        for (final java.util.List<TopologyElement> bucket : rowBuckets.values())
        {
            final int baseIndent = rowIndent[rIdx++];
            final double rowXmin;
            {
                double m = Double.POSITIVE_INFINITY;
                for (final TopologyElement el : bucket) m = Math.min(m, el.centroid().getX());
                rowXmin = m;
            }
            // Find this bucket's grid row index. rowBuckets iterates
            // reverse-y; we want top row at grid[0].
            final int row = rIdx - 1;
            for (final TopologyElement el : bucket)
            {
                final double x = el.centroid().getX();
                final int cellIdx = (int) Math.round((x - rowXmin) / ux);
                final int tc = baseIndent + cellIdx * textColsPerCell;
                if (tc < 0 || tc + glyphSlots > grid[row].length) continue;
                final int idx = el.index();
                final int who = cs.who(idx, siteType);
                if (who <= 0)
                {
                    grid[row][tc] = EMPTY;
                    if (hasDepth) grid[row][tc + 1] = STACK_PAD;
                }
                else
                {
                    grid[row][tc] = pickGlyph(gm, cs, idx, siteType);
                    if (hasDepth) grid[row][tc + 1] = depthChar(cs, idx, siteType);
                }
            }
        }
        for (int r = 0; r < rows; r++)
        {
            int end = grid[r].length;
            while (end > 0 && grid[r][end - 1] == OFFBOARD) end--;
            out.append(new String(grid[r], 0, end)).append('\n');
        }
    }

    private static void appendCompact(
        final StringBuilder out, final Context ctx, final GlyphMap gm,
        final List<? extends TopologyElement> sites, final SiteType siteType,
        final ContainerState cs)
    {
        out.append("[irregular layout — occupied sites listed; full geometry in FACTS]\n");
        final java.util.List<String> entries = new java.util.ArrayList<>();
        for (final TopologyElement el : sites)
        {
            final int idx = el.index();
            final int who = cs.who(idx, siteType);
            if (who <= 0) continue;
            final char g = pickGlyph(gm, cs, idx, siteType);
            final int depth = Math.max(cs.sizeStack(idx, siteType), cs.count(idx, siteType));
            final String tag = depth > 1 ? (g + Integer.toString(depth)) : Character.toString(g);
            entries.add(el.label() + ":" + tag);
        }
        entries.sort(String::compareTo);
        if (entries.isEmpty())
        {
            out.append("(empty)\n");
            return;
        }
        // Wrap at ~80 chars for readability.
        int lineLen = 0;
        for (int i = 0; i < entries.size(); i++)
        {
            final String e = entries.get(i);
            if (lineLen > 0 && lineLen + 2 + e.length() > 80)
            {
                out.append('\n');
                lineLen = 0;
            }
            if (lineLen > 0) { out.append("  "); lineLen += 2; }
            out.append(e);
            lineLen += e.length();
        }
        out.append('\n');
    }

    private static double withinRowUx(
        final java.util.SortedMap<Long, java.util.List<TopologyElement>> rowBuckets)
    {
        final java.util.ArrayList<Double> gaps = new java.util.ArrayList<>();
        for (final java.util.List<TopologyElement> bucket : rowBuckets.values())
        {
            if (bucket.size() < 2) continue;
            final double[] xs = new double[bucket.size()];
            for (int i = 0; i < bucket.size(); i++) xs[i] = bucket.get(i).centroid().getX();
            Arrays.sort(xs);
            for (int i = 1; i < xs.length; i++)
            {
                final double d = xs[i] - xs[i - 1];
                if (d > 1e-9) gaps.add(d);
            }
        }
        if (gaps.isEmpty())
        {
            // Fall back to overall median x-gap if no row has >= 2 sites.
            return Math.max(1.0, 0.0);
        }
        java.util.Collections.sort(gaps);
        return gaps.get(gaps.size() / 2);
    }

    private static boolean anyDepth(final Context ctx)
    {
        final ContainerState[] cstates = ctx.state().containerStates();
        final Container[] containers = ctx.containers();
        for (int ci = 0; ci < containers.length; ci++)
        {
            final SiteType st = containers[ci].defaultSite();
            final ContainerState cs = cstates[ci];
            for (final TopologyElement el : siteElements(ctx.game(), containers[ci], st))
            {
                final int idx = el.index();
                if (cs.sizeStack(idx, st) > 1) return true;
                if (cs.count(idx, st) > 1) return true;
            }
        }
        return false;
    }

    private static char pickGlyph(
        final GlyphMap gm, final ContainerState cs, final int site, final SiteType st)
    {
        final int who;
        final int what;
        if (cs.sizeStack(site, st) > 1)
        {
            // Show the top piece's identity.
            final int top = cs.sizeStack(site, st) - 1;
            who = cs.who(site, top, st);
            what = cs.what(site, top, st);
        }
        else
        {
            who = cs.who(site, st);
            what = cs.what(site, st);
        }
        if (who <= 0) return EMPTY;
        final Character g = gm.charByKey.get(key(who, what));
        if (g != null) return g;
        if (who < PLAYER_GLYPHS.length) return PLAYER_GLYPHS[who];
        throw new IllegalStateException("no glyph for owner=" + who + " what=" + what);
    }

    private static char depthChar(final ContainerState cs, final int site, final SiteType st)
    {
        final int sz = cs.sizeStack(site, st);
        final int ct = cs.count(site, st);
        final int depth = Math.max(sz, ct);
        if (depth <= 1) return STACK_PAD;
        if (depth <= 9) return (char) ('0' + depth);
        if (depth <= 35) return (char) ('a' + (depth - 10));
        return '+';
    }

    // ------------------------------------------------------- glyph map

    private static final class GlyphMap
    {
        // key = (owner << 32) | (what & 0xFFFFFFFFL)
        final Map<Long, Character> charByKey = new HashMap<>();
        final List<String> legend = new ArrayList<>();
    }

    private static long key(final int owner, final int what)
    {
        return ((long) owner << 32) | (what & 0xFFFFFFFFL);
    }

    private static GlyphMap buildGlyphMap(final Context ctx)
    {
        final Component[] components = ctx.components();
        final Container[] containers = ctx.containers();
        final ContainerState[] cstates = ctx.state().containerStates();

        // owner -> (what -> firstSeenName) — TreeMap for deterministic order.
        final TreeMap<Integer, TreeMap<Integer, String>> typesByOwner = new TreeMap<>();
        for (int ci = 0; ci < containers.length; ci++)
        {
            final Container cont = containers[ci];
            final SiteType st = cont.defaultSite();
            final ContainerState cs = cstates[ci];
            for (final TopologyElement el : siteElements(ctx.game(), cont, st))
            {
                final int idx = el.index();
                final int who = cs.who(idx, st);
                if (who <= 0) continue;
                final int sz = cs.sizeStack(idx, st);
                if (sz > 1)
                {
                    for (int lv = 0; lv < sz; lv++)
                        recordType(typesByOwner, cs.who(idx, lv, st), cs.what(idx, lv, st), components);
                }
                else
                {
                    recordType(typesByOwner, who, cs.what(idx, st), components);
                }
            }
        }

        final GlyphMap gm = new GlyphMap();
        // Global taken set so glyphs never collide across players.
        // (A 4-player game with multiple piece types per player can otherwise
        // have P1 'C' and P3 'C' both map to "Counter".)
        final java.util.Set<Character> taken = new java.util.HashSet<>();
        final java.util.Set<Integer> collapsedOwners = new java.util.LinkedHashSet<>();
        for (final Map.Entry<Integer, TreeMap<Integer, String>> e : typesByOwner.entrySet())
        {
            final int owner = e.getKey();
            final TreeMap<Integer, String> byWhat = e.getValue();
            if (byWhat.size() == 1)
            {
                final Map.Entry<Integer, String> only = byWhat.firstEntry();
                final char og = ownerGlyph(owner);
                final char g;
                if (!taken.contains(og))
                {
                    g = og;
                }
                else
                {
                    final Character lg = tryPickLetterGlyph(only.getValue(), owner, taken);
                    g = (lg != null) ? lg : og;
                    if (lg == null) collapsedOwners.add(owner);
                }
                taken.add(g);
                gm.charByKey.put(key(owner, only.getKey()), g);
                gm.legend.add(g + " = P" + owner + " " + only.getValue());
            }
            else
            {
                final LinkedHashMap<Integer, String> ordered = new LinkedHashMap<>();
                byWhat.entrySet().stream()
                    .sorted(Map.Entry.comparingByValue())
                    .forEach(en -> ordered.put(en.getKey(), en.getValue()));
                for (final Map.Entry<Integer, String> en : ordered.entrySet())
                {
                    Character g = tryPickLetterGlyph(en.getValue(), owner, taken);
                    if (g == null)
                    {
                        g = ownerGlyph(owner);
                        collapsedOwners.add(owner);
                    }
                    taken.add(g);
                    gm.charByKey.put(key(owner, en.getKey()), g);
                    gm.legend.add(g + " = P" + owner + " " + en.getValue());
                }
            }
        }
        for (final int owner : collapsedOwners)
            gm.legend.add("(P" + owner + " piece types collapsed onto '"
                + ownerGlyph(owner) + "' — see FACTS)");
        gm.legend.sort(String::compareTo);
        return gm;
    }

    private static void recordType(
        final TreeMap<Integer, TreeMap<Integer, String>> map,
        final int owner, final int what, final Component[] components)
    {
        if (owner <= 0) return;
        map.computeIfAbsent(owner, k -> new TreeMap<>())
           .putIfAbsent(what, pieceName(components, what));
    }

    private static char ownerGlyph(final int owner)
    {
        if (owner > 0 && owner < PLAYER_GLYPHS.length) return PLAYER_GLYPHS[owner];
        throw new IllegalStateException("no owner glyph for player " + owner);
    }

    private static Character tryPickLetterGlyph(
        final String pieceName, final int owner, final java.util.Set<Character> taken)
    {
        final boolean upper = (owner == 1) || (owner >= 3 && owner % 2 == 1);
        for (int i = 0; i < pieceName.length(); i++)
        {
            final char raw = pieceName.charAt(i);
            if (!Character.isLetter(raw)) continue;
            final char cand = upper ? Character.toUpperCase(raw) : Character.toLowerCase(raw);
            if (!taken.contains(cand)) return cand;
        }
        for (char c = upper ? 'A' : 'a'; c <= (upper ? 'Z' : 'z'); c++)
            if (!taken.contains(c)) return c;
        return null;
    }

    // ----------------------------------------------------------- helpers

    private static String pieceName(final Component[] components, final int what)
    {
        if (what <= 0 || what >= components.length) return "Piece";
        final Component c = components[what];
        if (c == null) return "Piece";
        String n = c.getNameWithoutNumber();
        if (n == null || n.isEmpty()) n = c.name();
        if (n == null || n.isEmpty()) n = "Piece";
        return n;
    }

    private static String qualifiedLabel(final int ci, final Container cont, final TopologyElement el)
    {
        final String raw = el.label();
        final String base = (raw == null || raw.isEmpty())
            ? (cont.name() + "-" + el.index()) : raw;
        if (ci == 0) return base;
        final String cname = cont.name();
        return ((cname == null || cname.isEmpty()) ? ("Container" + ci) : cname) + ":" + base;
    }

    private static double axisUnit(
        final List<? extends TopologyElement> sites, final boolean xAxis)
    {
        // Median of the non-trivial gaps between sorted axis values.
        // Median is robust to both noise (a few tiny near-zero gaps from
        // floating-point coincidences) and outliers (a few large gaps
        // from disconnected components). For regular boards every gap
        // equals the cell spacing so median == that spacing.
        final int n = sites.size();
        if (n < 2) return 0.0;
        final double[] vals = new double[n];
        double vmin = Double.POSITIVE_INFINITY, vmax = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < n; i++)
        {
            final Point2D c = sites.get(i).centroid();
            final double v = xAxis ? c.getX() : c.getY();
            vals[i] = v;
            if (v < vmin) vmin = v;
            if (v > vmax) vmax = v;
        }
        Arrays.sort(vals);
        final double span = vmax - vmin;
        final double tol = Math.max(1e-9, span * 1e-4);
        final java.util.ArrayList<Double> gaps = new java.util.ArrayList<>();
        for (int i = 1; i < vals.length; i++)
        {
            final double d = vals[i] - vals[i - 1];
            if (d > tol) gaps.add(d);
        }
        if (gaps.isEmpty()) return Math.max(span, 1e-6);
        java.util.Collections.sort(gaps);
        return gaps.get(gaps.size() / 2);
    }

    @SuppressWarnings("unchecked")
    private static List<? extends TopologyElement> siteElements(
        final Game game, final Container cont, final SiteType siteType)
    {
        switch (siteType)
        {
            case Vertex: return cont.topology().vertices();
            case Edge:   return cont.topology().edges();
            case Cell:   default: return cont.topology().cells();
        }
    }
}
