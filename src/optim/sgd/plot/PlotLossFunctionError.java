package optim.sgd.plot;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.Rectangle;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import javax.swing.JFrame;
import org.jfree.chart.ChartRenderingInfo;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotRenderingInfo;
import org.jfree.chart.plot.PlotState;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

/**
 *
 * @author diego_paez
 */
public class PlotLossFunctionError extends JFrame{

    private XYSeries seriesActual;
    private XYPlot plot;
    private int currVal;
    private long timeAtLastRepaint;

    public PlotLossFunctionError(double stepSize) {
        NumberFormat formatter = new DecimalFormat("#0.00");  

        // Create plot series
        seriesActual = new XYSeries("LF w/step=" + formatter.format(stepSize));
        currVal = 1;

        // Add series to collection
        XYSeriesCollection allSeries = new XYSeriesCollection();
        allSeries.addSeries(seriesActual);

        // Create renderer for plot
        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();

        // remove dots from first series
        renderer.setSeriesShapesVisible(0, false);
    
        // Create plot - the plot is called in the paint method, i.e., when hr is set visible
        plot = new XYPlot(allSeries, new NumberAxis(), new NumberAxis(),
                renderer);
        
        setBackground(Color.white);
        setSize(600,600);
        setBackground(Color.WHITE);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        timeAtLastRepaint = System.currentTimeMillis();

    }

    public void addData(double d){
        // Fill data for plotting series
        seriesActual.add((double) currVal, d);
        long currTime = System.currentTimeMillis();
        if(currTime - timeAtLastRepaint > 500){
            repaint();
            timeAtLastRepaint = System.currentTimeMillis();
        }
        currVal++;
    }

    public void paint(Graphics g){
        Rectangle rect = new Rectangle(590,590);
        Point anchor = new Point(100,150);
        PlotState ps = new PlotState();
        PlotRenderingInfo pr = new PlotRenderingInfo(new ChartRenderingInfo());

        g.setColor(Color.WHITE);
        g.fillRect(0, 0, 600, 600);
        plot.draw((Graphics2D) g, rect, anchor, ps, pr);
    }
}