import java.awt.geom.Point2D;
import java.io.File;
import java.util.TreeSet;

import game.Game;
import other.GameLoader;
import other.topology.TopologyElement;
import game.equipment.container.Container;

public class ProbeCentroids
{
    public static void main(final String[] args)
    {
        final Game g = GameLoader.loadGameFromFile(new File(args[0]));
        final Container board = g.equipment().containers()[0];
        final java.util.List<? extends TopologyElement> sites;
        switch (board.defaultSite())
        {
            case Vertex: sites = board.topology().vertices(); break;
            case Edge:   sites = board.topology().edges(); break;
            default:     sites = board.topology().cells(); break;
        }
        final TreeSet<Double> ys = new TreeSet<>();
        for (final TopologyElement el : sites)
        {
            final Point2D c = el.centroid();
            ys.add(Math.round(c.getY() * 1e6) / 1e6);
        }
        System.out.println("sites=" + sites.size() + " distinctY=" + ys.size());
        for (final TopologyElement el : sites)
        {
            final Point2D c = el.centroid();
            System.out.printf("  %s  (%.3f, %.3f)%n", el.label(), c.getX(), c.getY());
        }
        System.out.println("---ys---");
        Double prev = null;
        for (final Double y : ys)
        {
            if (prev != null) System.out.printf("  y=%.6f  gap=%.6f%n", y, y - prev);
            else              System.out.printf("  y=%.6f%n", y);
            prev = y;
        }
    }
}
