import javafx.stage.Screen;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ComponentEvent;
import java.awt.event.ComponentListener;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Main {

   //Declare variables
   public static ArrayList<DataSet> userInputData = new ArrayList();
    public static ArrayList<Integer> exclusionList = new ArrayList();
   static DV dv;
    static DV oldDV;
    static double accuracy;
   static boolean domainActive = true;
   static boolean groupingActive = true;
   static boolean needSplitting = true;
   static boolean hasIndex;
    static boolean hasClass;
    static boolean[] isMirrored;
   static double domainMinPercent = 0;
   static double domainMaxPercent = 100;
   static double domainMinValue;
   static double domainMaxValue;
   static double sortingLinesPercent[];
   static double sortingLinesValue[];

    /**
     * Main method of the program.
     * Will create the main UI components and manage their operation
     * @param args standard main method command line arguments.
     * @throws IOException
     */
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
       UIManager.setLookAndFeel(
                UIManager.getSystemLookAndFeelClassName());
        Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
        if(screenSize.getWidth() >= 1280)
        {
            if(screenSize.getWidth() >= 1920)
            {
                Resolutions.setResolution(0);
            }
            else if(screenSize.getHeight() >= 900)
            {

            }
            else
            {
                Resolutions.setResolution(1);
            }
        }
		dv = new DV();
		dv.setVisible(true);
	}

}
