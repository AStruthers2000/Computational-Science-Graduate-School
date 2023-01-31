//Author:Fawziah Alkharnda
//DV Program

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.DefaultDrawingSupplier;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYBarRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.*;
import org.jfree.ui.RectangleInsets;
import slider.RangeSlider;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.StringTokenizer;

import static javax.swing.ScrollPaneConstants.HORIZONTAL_SCROLLBAR_NEVER;


public class DV extends JFrame implements DocumentListener {

    public static int currentGroupingLine = 0;
    RangeSlider domainSlider2;

    public static double[] initialAngles;

    private static long seriesCount = 0;
    private static final long serialVersionUID = 1L;

    JFrame window = this;

    //Declaration for the textArea
    JPanel sliderPanel;
    JPanel jp = new JPanel();
    JPanel sortingLabel;
    JPanel graphPanel;
    JPanel graphDomainPanel;
    ChartPanel tempoPanel;
    JTextArea analyticsText;
    slider.RangeSlider domainSlider;

    //Variables to handle threshold panel
    boolean loadNewRanges = true;
    boolean openRanges = false;
    boolean spinnerFlag;
    String splittableRanges;

    //Threshold color variables
    static Color activeThresholdLine = new Color(0, 160, 0);
    static Color inactiveThresholdLine = new Color(160, 0, 0);

    //Boolean to control basr graphs
    boolean isBars = false;

    //Create the three main scroll panes that will be used in the program
    JScrollPane graphPane, anglesPane, analyticsPane;

    Range ranges;

    /**
     * Main handler for UI.
     */
    public DV() {
        super("DV Program");
        this.setName("DV Program");
        this.setSize(Resolutions.dvWindow[0], Resolutions.dvWindow[1]);
        this.setLocationRelativeTo(null);
        this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        this.setVisible(true);
        this.setExtendedState(JFrame.NORMAL);
        this.setExtendedState(this.getExtendedState() & (~JFrame.ICONIFIED));
        this.setResizable(false);

        //Create the menu
        createMenuBar(this);

        //Create the toolbar  and add buttons
        JToolBar toolBar = new JToolBar("Tool Palette");
        toolBar.setFloatable(false);
        toolBar.setRollover(true);

        JFrame frame = this;

        JButton newButton = new JButton("New");
        newButton.setToolTipText("Create a new project");
        newButton.addActionListener( new ActionListener()
        {
            public void actionPerformed(ActionEvent e) {
                //Create code for file opener
                JFileChooser fc = new JFileChooser();
                FileNameExtensionFilter filter = new FileNameExtensionFilter("csv", "csv");
                fc.setFileFilter(filter);

                loadNewRanges = true;
                //Find out if there is an index
                Main.hasIndex = false;
                int n = JOptionPane.showConfirmDialog(
                        frame,
                        "Does this project use column 0 as the index?",
                        "Index",
                        JOptionPane.YES_NO_OPTION);

                if(n == 0)
                {
                    Main.hasIndex = true;
                }

                //Find out if it has built in classes
                Main.hasClass = false;
                n = JOptionPane.showConfirmDialog(
                        frame,
                        "Does this project use the last column to designate classes?",
                        "Classes",
                        JOptionPane.YES_NO_OPTION);

                if(n == 0)
                {
                    Main.hasClass = true;
                    Main.needSplitting = true;
                }

                //Open the file chooser dialog and get the users file location

                int returnVal = fc.showOpenDialog(DV.this);

                //Check if the return value is a valid path
                if (returnVal == JFileChooser.APPROVE_OPTION) {
                    //get the file that the user selected
                    File inputFile = fc.getSelectedFile();
                    //Clear the current project
                    Main.userInputData.clear();
                    DVParser.isFirst = true;
                    DVParser.minValues.clear();
                    DVParser.maxValues.clear();
                    DVParser.files.clear();
                    //Create a new parser and parse the file
                    DVParser inputParser = new DVParser();
                    try {
                        inputParser.parseData(inputFile);
                    }
                    catch(Exception exc)
                    {
                        JOptionPane.showConfirmDialog(
                                frame,
                                "Error opening target file.",
                                "Error",
                                JOptionPane.YES_OPTION);
                    }
                    Range.ranges.clear();
                    createSortingPanel();
                    //Update the slider panel to fit all the sliders
                    sliderPanel.setPreferredSize(new Dimension(Resolutions.sliderPanel[0],(100 * Main.userInputData.get(0).fieldAngles.length)));

                    //Set initial grouping line conditions.
                    //Main.sortingLinesPercent = new double[Main.userInputData.size() - 1];
                    //Main.sortingLinesValue = new double[Main.userInputData.size() - 1];

                    AdditionalOptions.updateSorting();
                    currentGroupingLine = 0;
                    if(Main.userInputData.size() > 1)
                    {
                        //domainSlider2.setValue((int)Main.sortingLinesPercent[currentGroupingLine]);
                    }

                    //Draw the graph
                    AdditionalOptions.updateDomains();
                    drawGraphs();
                }
                //Update the field angles, then redraw UI
                updateFieldAngles(sliderPanel);
                repaint();
                revalidate();
            }
        });
        toolBar.add(newButton);

        JButton importButton = new JButton("Import");
        importButton.setToolTipText("Import additional data into your project.");
        importButton.addActionListener( new ActionListener()
        {
            public void actionPerformed(ActionEvent e) {
                loadNewRanges = true;
                //Open the file chooser dialog and get the users file location
                //if(Main.hasClass)
                {
                    //JOptionPane.showMessageDialog(frame,
                    //       "Multiple files are not supported when classes are enabled",
                    //     "Error",
                    //   JOptionPane.ERROR_MESSAGE);
                    //}
                    // else {
                    FileNameExtensionFilter filter = new FileNameExtensionFilter("csv", "csv");
                    JFileChooser fc = new JFileChooser();
                    fc.setFileFilter(filter);
                    int returnVal = fc.showOpenDialog(DV.this);

                    //Check if the return value is a valid path
                    if (returnVal == JFileChooser.APPROVE_OPTION) {
                        //get the file that the user selected
                        File inputFile = fc.getSelectedFile();
                        //Create a new parser and parse the file
                        DVParser inputParser = new DVParser();
                        try {
                            inputParser.parseData(inputFile);
                        }
                        catch(Exception exc)
                        {
                            JOptionPane.showConfirmDialog(
                                    frame,
                                    "Error opening target file.",
                                    "Error",
                                    JOptionPane.YES_OPTION);
                        }
                        //Update the slider panel to fit all the sliders
                        //Set initial grouping line conditions.
                        Main.sortingLinesPercent = new double[Main.userInputData.size() - 1];
                        Main.sortingLinesValue = new double[Main.userInputData.size() - 1];
                        for(int i = 0; i < Main.sortingLinesPercent.length; i++)
                        {
                            Main.sortingLinesPercent[i] = i;
                        }

                        if(Main.userInputData.size() > 1)
                        {
                            // domainSlider2.setValue((int)Main.sortingLinesPercent[currentGroupingLine]);
                        }
                    }
                    loadNewRanges = true;
                    Range.ranges.clear();
                    createSortingPanel();
                    AdditionalOptions.updateSorting();
                    currentGroupingLine = 0;
                    //Draw the graph
                    AdditionalOptions.updateDomains();
                    drawGraphs();
                    //Update the field angles, then redraw UI
                    updateFieldAngles(sliderPanel);
                    repaint();
                    revalidate();
                }
            }
        });
        toolBar.add(importButton);

        JButton openButton = new JButton("Open");
        openButton.setToolTipText("Open a saved project");
        openButton.addActionListener( new ActionListener()
        {
            public void actionPerformed(ActionEvent e) {
                loadNewRanges = true;
                //Open the file chooser dialog and get the users file location
                FileNameExtensionFilter filter = new FileNameExtensionFilter("datv", "datv");
                JFileChooser fc = new JFileChooser();
                fc.setFileFilter(filter);
                int returnVal = fc.showOpenDialog(DV.this);
                try {
                    if (returnVal == JFileChooser.APPROVE_OPTION) {
                        //get the file that the user selected
                        File inputFile = fc.getSelectedFile();
                        SaveCreator.loadSave(inputFile);
                    }
                }
                catch(Exception ex)
                {
                    JOptionPane.showMessageDialog(frame,
                            "Error loading File");
                }
                sliderPanel.removeAll();
                //Update the slider panel to fit all the sliders
                sliderPanel.setPreferredSize(new Dimension(Resolutions.sliderPanel[0],(100 * SaveCreator.projectAngles.length)));

                AdditionalOptions.updateSorting();
                currentGroupingLine = 0;
                AdditionalOptions.updateDomains();
                openRanges = true;
                createSortingPanel();

                //Update the field angles, then redraw UI
                for(int i = 0; i < Main.userInputData.get(0).fieldLength; i++)
                {
                    sliderPanel.setLayout(new GridLayout(Main.userInputData.get(0).fields.length, 0));
                    String fieldName = Main.userInputData.get(0).getFieldName(i);
                    int fieldAngle = SaveCreator.projectAngles[i];
                    Main.userInputData.get(0).fieldAngles[i] = fieldAngle;
                    SliderWithIntegration.createPanel(fieldName, fieldAngle, i, sliderPanel);
                }
                drawGraphs();
                repaint();
                revalidate();

            }
        });
        toolBar.add(openButton);

        JButton saveButton = new JButton("Save");
        saveButton.setToolTipText("Save a project to use later");
        saveButton.addActionListener( new ActionListener()
        {
            public void actionPerformed(ActionEvent e) {
                FileNameExtensionFilter filter = new FileNameExtensionFilter("datv", "datv");
                JFileChooser fc = new JFileChooser();
                fc.setFileFilter(filter);
                if(SaveCreator.currentProject == null)
                {
                    //Open the file chooser dialog and get the users file location
                    int returnVal = fc.showSaveDialog(DV.this);
                    //Check if the return value is a valid path
                    try {
                        if (returnVal == JFileChooser.APPROVE_OPTION) {
                            //get the file that the user selected
                            File inputFile = new File(fc.getSelectedFile().getPath() + ".datv");
                            SaveCreator.saveAs(inputFile);
                        }
                    }
                    catch(Exception ex)
                    {
                        JOptionPane.showMessageDialog(frame,
                                "Error Saving File");
                    }
                }
                else
                {
                    try {
                        SaveCreator.save();
                    }
                    catch(Exception E)
                    {
                        JOptionPane.showMessageDialog(frame,
                                "Error Saving File");
                    }
                }
            }
        });
        toolBar.add(saveButton);

        JLabel seperator = new JLabel("  |  ");
        toolBar.add(seperator);

        JButton optionsButton = new JButton("Options");
        optionsButton.setToolTipText("Open the options menu");
        optionsButton.addActionListener( new ActionListener()
        {
            public void actionPerformed(ActionEvent e) {
                //Open the options menu
                OptionsMenu menu = new OptionsMenu();
            }
        });
        toolBar.add(optionsButton);

        JButton unzoomButton = new JButton("Reset Screen");
        unzoomButton.setToolTipText("Resets rendered zoom area");
        unzoomButton.addActionListener( new ActionListener()
        {
            @Override
            public void actionPerformed(ActionEvent e)
            {
                drawGraphs();
            }
        });
        toolBar.add(unzoomButton);

        JButton optButton = new JButton("Optimize Angles");
        optButton.setToolTipText("Attempts to find optimization fof angles using current thresholds");
        optButton.addActionListener( new ActionListener()
        {
            public void actionPerformed(ActionEvent e) {
                Random rand = new Random(System.currentTimeMillis());
                sliderPanel.removeAll();
                int count = 0;
                boolean foundBetter = false;
                initialAngles = Arrays.copyOf(Main.userInputData.get(0).fieldAngles, Main.userInputData.get(0).fieldLength);
                double initialAccuracy = Main.accuracy;
                while(count < 1000) {
                    sliderPanel.removeAll();
                    for (int i = 0; i < Main.userInputData.get(0).fieldLength; i++) {
                        sliderPanel.setLayout(new GridLayout(Main.userInputData.get(0).fields.length, 0));
                        String fieldName = Main.userInputData.get(0).getFieldName(i);
                        int fieldAngle = rand.nextInt(181);
                        Main.userInputData.get(0).fieldAngles[i] = fieldAngle;
                        SliderWithIntegration.createPanel(fieldName, fieldAngle, i, sliderPanel);
                    }
                    drawGraphs();
                    count++;
                    if(Main.accuracy > initialAccuracy)
                    {
                        count = 1001;
                        foundBetter = true;
                    }
                }

                if(foundBetter == false)
                {
                    JOptionPane.showMessageDialog(
                            frame,
                            "Was unable to optimize angles.\n",
                            "Error",
                            JOptionPane.OK_OPTION);
                    sliderPanel.removeAll();
                    for (int i = 0; i < Main.userInputData.get(0).fieldLength; i++) {
                        sliderPanel.setLayout(new GridLayout(Main.userInputData.get(0).fields.length, 0));
                        String fieldName = Main.userInputData.get(0).getFieldName(i);
                        int fieldAngle = (int)initialAngles[i];
                        Main.userInputData.get(0).fieldAngles[i] = fieldAngle;
                        SliderWithIntegration.createPanel(fieldName, fieldAngle, i, sliderPanel);
                    }
                }

                drawGraphs();
                repaint();
                revalidate();

            }

        });
        toolBar.add(optButton);

        JButton unoptButton = new JButton("Undo Optimization");
        unoptButton.setToolTipText("Reverses previous operation");
        unoptButton.addActionListener( new ActionListener()
        {
            public void actionPerformed(ActionEvent e) {
                sliderPanel.removeAll();
                for (int i = 0; i < Main.userInputData.get(0).fieldLength; i++) {
                    sliderPanel.setLayout(new GridLayout(Main.userInputData.get(0).fields.length, 0));
                    String fieldName = Main.userInputData.get(0).getFieldName(i);
                    int fieldAngle = (int)initialAngles[i];
                    Main.userInputData.get(0).fieldAngles[i] = fieldAngle;
                    SliderWithIntegration.createPanel(fieldName, fieldAngle, i, sliderPanel);

                }

                drawGraphs();
                repaint();
                revalidate();
            }
        });
        toolBar.add(unoptButton);

        JButton barButton = new JButton("Toggle Barline");
        barButton.setToolTipText("Toggles graph showing barline of endpoint placement");
        barButton.addActionListener( new ActionListener()
        {
            public void actionPerformed(ActionEvent e) {
                isBars = !isBars;

                drawGraphs();
                repaint();
                revalidate();
            }
        });
        toolBar.add(barButton);

        JLabel seperatorTwo = new JLabel("  |  ");
        toolBar.add(seperatorTwo);

        JButton helpButton = new JButton("Manual");
        helpButton.setToolTipText("Opens user manual");
        helpButton.addActionListener( new ActionListener()
        {
            public void actionPerformed(ActionEvent e) {
                try {
                    File file = new File(System.getProperty("user.dir") + "\\DVHelp.pdf");
                    Desktop desktop = Desktop.getDesktop();
                    desktop.open(file);
                }
                catch(Exception ex)
                {
                    JOptionPane.showMessageDialog(
                            frame,
                            "Error opening help file.\n" + ex,
                            "Error",
                            JOptionPane.OK_OPTION);
                }
            }
        });
        toolBar.add(helpButton);

        JPanel toolbarPanel = new JPanel(new BorderLayout());

        toolbarPanel.add(toolBar, BorderLayout.PAGE_START);

        //Create the main panel for the program and set its constraints using gridbaglayout
        JPanel mainPanel = new JPanel(new GridBagLayout());
        GridBagConstraints constraints = new GridBagConstraints();

        //Code for creating initial graph
        XYSeriesCollection data = new XYSeriesCollection();
        JFreeChart chart = ChartFactory.createXYLineChart("", "", "", data);
        XYPlot plot = (XYPlot) chart.getPlot();
        plot.setDrawingSupplier(new DefaultDrawingSupplier(
                new Paint[] {Color.RED},
                DefaultDrawingSupplier.DEFAULT_OUTLINE_PAINT_SEQUENCE,
                DefaultDrawingSupplier.DEFAULT_STROKE_SEQUENCE,
                DefaultDrawingSupplier.DEFAULT_OUTLINE_STROKE_SEQUENCE,
                DefaultDrawingSupplier.DEFAULT_SHAPE_SEQUENCE));
        plot.getRangeAxis().setVisible(false);
        plot.getDomainAxis().setVisible(false);
        chart.removeLegend();
        plot.setRangeGridlinesVisible(false);
        chart.setBorderVisible(false);
        ChartPanel tempoPanel = new ChartPanel(chart);
        tempoPanel.setPreferredSize(new Dimension(Resolutions.tempoPanel[0], Resolutions.tempoPanel[1]));
        //Code for creating graph end

        graphPanel = new JPanel();
        BoxLayout gpLayout = new BoxLayout(graphPanel, BoxLayout.Y_AXIS);
        graphPanel.setLayout(gpLayout);
        graphPanel.add(tempoPanel);

        graphPane = new JScrollPane(graphPanel);
        graphPane.setAlignmentX(Component.CENTER_ALIGNMENT);
        graphPane.setHorizontalScrollBarPolicy(HORIZONTAL_SCROLLBAR_NEVER);
        constraints.weightx = 0.7;
        constraints.gridx = 0;
        constraints.gridy = 0;
        
        graphDomainPanel = new JPanel();
        graphDomainPanel.setLayout(new BoxLayout(graphDomainPanel, BoxLayout.Y_AXIS));
        graphDomainPanel.setPreferredSize(new Dimension(Resolutions.graphDomainPanel[0], Resolutions.graphDomainPanel[1]));
        graphDomainPanel.add(graphPane);
        
        //Create slider to handle range and grouping
        JPanel domainPanel = new JPanel();

        domainSlider = new RangeSlider();

        domainSlider.setMinimum(0);
        domainSlider.setMaximum(100);

        domainSlider.setValue(0);
        domainSlider.setUpperValue(100);

        domainSlider.setToolTipText("Control visible range of graph");

        // Add listener to update display.
        domainSlider.addChangeListener(new ChangeListener() {
            public void stateChanged(ChangeEvent e) {
                RangeSlider slider = (RangeSlider) e.getSource();
                Main.domainMinPercent = slider.getValue();
                Main.domainMaxPercent = slider.getUpperValue();
                AdditionalOptions.updateDomains();
                drawGraphs();
            }});

        domainSlider.setPreferredSize(new Dimension(Resolutions.domainSlider[0], Resolutions.domainSlider[1]));
        domainSlider.setAlignmentX(Component.CENTER_ALIGNMENT);
        
        domainPanel.add(domainSlider);
        
        graphDomainPanel.add(domainPanel);

        JPanel domainLabel = new JPanel();
        domainLabel.add(new JLabel("Range Control"));
        domainLabel.setToolTipText("Control visible range of graph");
        graphDomainPanel.add(domainLabel);

        mainPanel.add(graphDomainPanel, constraints);
        
        //Code for sorting slider
        JPanel domainPanel2 = new JPanel();
        domainSlider2 = new RangeSlider();
        domainSlider2.setMinimum(0);
        domainSlider2.setMaximum(100);
        domainSlider2.setMajorTickSpacing(1);
        domainSlider2.setUpperValue(100);
        domainSlider2.setValue(0);
        domainSlider2.setToolTipText("Change threshold values for current threshold");

        domainSlider2.addChangeListener(new ChangeListener() {
            public void stateChanged(ChangeEvent e) {
                if(spinnerFlag)
                {
                    spinnerFlag = false;
                }
                else {
                    RangeSlider slider = (RangeSlider) e.getSource();
                    AdditionalOptions.updateDomains();
                    Range.ranges.get(currentGroupingLine).setRange(slider.getValue(), slider.getUpperValue());
                    drawGraphs();
                    repaint();
                    revalidate();
                }
            }});
        
        domainSlider2.setPreferredSize(new Dimension(Resolutions.domainSlider[0], Resolutions.domainSlider[1]));
        domainSlider2.setAlignmentX(Component.CENTER_ALIGNMENT);
        domainPanel2.add(domainSlider2);

        //Finalize domain panel
        graphDomainPanel.add(domainPanel2);
        mainPanel.add(graphDomainPanel, constraints);

        //Add grouping line spinner
        sortingLabel = new JPanel();
        createSortingPanel();
        graphDomainPanel.add(sortingLabel);

        //Create the angles list and add it to the corresponding scroll pane
        sliderPanel = new JPanel(new GridLayout(1,0));
        sliderPanel.setPreferredSize(new Dimension(Resolutions.sliderPanel[0], Resolutions.sliderPanel[1]));
        anglesPane = new JScrollPane(sliderPanel);
        anglesPane.setPreferredSize(new Dimension(Resolutions.anglesPane[0], Resolutions.anglesPane[1]));
        anglesPane.setHorizontalScrollBarPolicy(HORIZONTAL_SCROLLBAR_NEVER);
        constraints.weightx = 0.3;
        constraints.gridx = 1;
        constraints.gridy = 0;

        mainPanel.add(anglesPane,constraints);


        //Create the analytics text and add it to the corresponding scroll pane
        analyticsText = new JTextArea(10, 200);
        jp.add(analyticsText);
        analyticsPane = new JScrollPane(jp);
        analyticsPane.setPreferredSize(new Dimension(Resolutions.analyticsPane[0], Resolutions.analyticsPane[1]));
        analyticsPane.setHorizontalScrollBarPolicy(HORIZONTAL_SCROLLBAR_NEVER);
        constraints.weightx = 1;
        constraints.gridx = 0;
        constraints.gridy = 1;
        constraints.gridwidth = 2;
        mainPanel.add(analyticsPane, constraints);

        toolbarPanel.add(mainPanel, BorderLayout.CENTER);
        add(toolbarPanel);
        setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        //pack();
        setVisible(true);
    }

    @Override
    public void insertUpdate(DocumentEvent e) {
        // TODO Auto-generated method stub

    }


    @Override
    public void removeUpdate(DocumentEvent e) {
        // TODO Auto-generated method stub

    }


    @Override
    public void changedUpdate(DocumentEvent e) {
        // TODO Auto-generated method stub

    }

    /**
     * Updates field angles and generates array of sliders to control them. Generates new angles
     * @param sliderPanel Panel to stor array of sliders
     */
    private static void updateFieldAngles(JPanel sliderPanel) {
        sliderPanel.removeAll();
        sliderPanel.setLayout(new GridLayout(Main.userInputData.get(0).fields.length, 0));
        int fieldNumber = Main.userInputData.get(0).fields.length;
        for (int j = 0; j < fieldNumber; j++) {
            String fieldName = Main.userInputData.get(0).getFieldName(j);
            int fieldAngle = 45;
            SliderWithIntegration.createPanel(fieldName, fieldAngle, j, sliderPanel);
        }
    }

    /**
     * Updates field angles and generates array of sliders to control them
     * @param sliderPanel Panel to store array of sliders
     * @param flag Flag to indicate not to generate new angles
     */
    private static void updateFieldAngles(JPanel sliderPanel, int flag) {
        sliderPanel.removeAll();
        sliderPanel.setLayout(new GridLayout(Main.userInputData.get(0).fields.length, 0));
        int fieldNumber = Main.userInputData.get(0).fieldLength;
        for (int j = 0; j < fieldNumber - 1; j++) {
            String fieldName = Main.userInputData.get(0).getFieldName(j);
            int fieldAngle = SaveCreator.projectAngles[j];
            SliderWithIntegration.createPanel(fieldName, fieldAngle, j, sliderPanel);
        }
    }

    /**
     * Creates and handles threshold control panel
     */
    private void createSortingPanel()
    {
        sortingLabel.removeAll();
        if(loadNewRanges) {
            Range.ranges.clear();
            for (int i = 0; i < Main.userInputData.size(); i++) {
                ranges = new Range(0, 100, i);
            }
            loadNewRanges = false;
        }
        else if(openRanges)
        {
            Range.ranges.clear();
            StringTokenizer rangeSplitter = new StringTokenizer(splittableRanges, ",", false);
            int j = 0;
            while(rangeSplitter.hasMoreTokens())
            {
                int max = Integer.parseInt(rangeSplitter.nextToken());
                int min = Integer.parseInt(rangeSplitter.nextToken());
                System.out.println(max + " " + min);
                ranges = new Range(min, max, j);
                j++;
            }
            openRanges = false;
        }
        int max = Range.ranges.size();
        if(max == 0)
        {
            max = 1;
        }
         SpinnerNumberModel groupingLineModel = new SpinnerNumberModel(currentGroupingLine+1, 1, max, 1);
        JLabel sLabel = new JLabel("Threshold Control");
        sLabel.setToolTipText("Control thresholds for confusion matrix.");
        JSpinner groupingLineSpinner = new JSpinner(groupingLineModel);
        groupingLineSpinner.setToolTipText("Change currently selected threshold");
        groupingLineSpinner.addChangeListener(new ChangeListener() {
            @Override
            public void stateChanged(ChangeEvent e) {
                currentGroupingLine = (Integer) groupingLineSpinner.getValue() - 1;
                AdditionalOptions.updateSorting();
                spinnerFlag = true;
                domainSlider2.setValue(Range.ranges.get(currentGroupingLine).minimumPercent);
                spinnerFlag = true;
                domainSlider2.setUpperValue(Range.ranges.get(currentGroupingLine).maximumPercent);
                spinnerFlag = true;
                domainSlider2.setValue(Range.ranges.get(currentGroupingLine).minimumPercent);
                //Range.ranges.get(currentGroupingLine).setRange(domainSlider2.getValue(), domainSlider2.getUpperValue());
                AdditionalOptions.updateSorting();
                drawGraphs();
            }

        });
        sortingLabel.removeAll();
        sortingLabel.add(sLabel);
        sortingLabel.add(groupingLineSpinner);
   }

    /**
     * Creates main program menu bar
     * @param frame Frame of main program
     */
    private void createMenuBar(JFrame frame) {
        //Adding menu bar to the app
        JMenuBar menuBar = new JMenuBar();
        JMenu menu = new JMenu();
        JMenuItem menuItem = new JMenuItem();
        frame.setJMenuBar(menuBar);
        //Build the first menu.
        menu = new JMenu("File");
        menu.setMnemonic(KeyEvent.VK_A);
        menu.getAccessibleContext().setAccessibleDescription(
                "");
        menuBar.add(menu);

        //a group of JMenuItems
        JMenuItem menuItem0 = new JMenuItem("Create New Project",
                KeyEvent.VK_T);
        menuItem0.setAccelerator(KeyStroke.getKeyStroke(
                KeyEvent.VK_1, ActionEvent.ALT_MASK));
        menuItem0.getAccessibleContext().setAccessibleDescription(
                "");
        menu.add(menuItem0);

        //Create code for file opener
        JFileChooser fc = new JFileChooser();
        FileNameExtensionFilter filter = new FileNameExtensionFilter("csv", "csv");
        fc.setFileFilter(filter);

        menuItem0.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                loadNewRanges = true;
                //Find out if there is an index
                Main.hasIndex = false;
                int n = JOptionPane.showConfirmDialog(
                        frame,
                        "Does this project use column 0 as the index?",
                        "Index",
                        JOptionPane.YES_NO_OPTION);

                if(n == 0)
                {
                    Main.hasIndex = true;
                }

                //Find out if it has built in classes
                Main.hasClass = false;
                n = JOptionPane.showConfirmDialog(
                        frame,
                        "Does this project use the last column to designate classes?",
                        "Classes",
                        JOptionPane.YES_NO_OPTION);

                if(n == 0)
                {
                    Main.hasClass = true;
                    Main.needSplitting = true;
                }

                //Open the file chooser dialog and get the users file location

               int returnVal = fc.showOpenDialog(DV.this);

                //Check if the return value is a valid path
                if (returnVal == JFileChooser.APPROVE_OPTION) {
                    //get the file that the user selected
                    File inputFile = fc.getSelectedFile();
                    //Clear the current project
                    Main.userInputData.clear();
                    DVParser.isFirst = true;
                    DVParser.minValues.clear();
                    DVParser.maxValues.clear();
                    DVParser.files.clear();
                    //Create a new parser and parse the file
                    DVParser inputParser = new DVParser();
                    try {
                        inputParser.parseData(inputFile);
                    }
                    catch(Exception exc)
                    {
                        JOptionPane.showConfirmDialog(
                                frame,
                                "Error opening target file.",
                                "Error",
                                JOptionPane.YES_OPTION);
                    }
                    Range.ranges.clear();
                    createSortingPanel();
                    //Update the slider panel to fit all the sliders
                    sliderPanel.setPreferredSize(new Dimension(Resolutions.sliderPanel[0],(100 * Main.userInputData.get(0).fieldAngles.length)));

                    //Set initial grouping line conditions.
                    //Main.sortingLinesPercent = new double[Main.userInputData.size() - 1];
                    //Main.sortingLinesValue = new double[Main.userInputData.size() - 1];

                    AdditionalOptions.updateSorting();
                    currentGroupingLine = 0;
                    if(Main.userInputData.size() > 1)
                    {
                    //domainSlider2.setValue((int)Main.sortingLinesPercent[currentGroupingLine]);
                    }
                    
                    //Draw the graph
                    AdditionalOptions.updateDomains();
                    drawGraphs();
                }
                //Update the field angles, then redraw UI
                updateFieldAngles(sliderPanel);
                repaint();
                revalidate();
            }
        });

        JMenuItem menuItem1 = new JMenuItem("Open Saved Project",
                KeyEvent.VK_T);
        menuItem1.setAccelerator(KeyStroke.getKeyStroke(
                KeyEvent.VK_1, ActionEvent.ALT_MASK));
        menuItem1.getAccessibleContext().setAccessibleDescription(
                "");
        menu.add(menuItem1);

        menuItem1.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                loadNewRanges = true;
                //Open the file chooser dialog and get the users file location
                FileNameExtensionFilter filter = new FileNameExtensionFilter("datv", "datv");
                JFileChooser fc = new JFileChooser();
                fc.setFileFilter(filter);
                int returnVal = fc.showOpenDialog(DV.this);
                try {
                    if (returnVal == JFileChooser.APPROVE_OPTION) {
                        //get the file that the user selected
                        File inputFile = fc.getSelectedFile();
                        SaveCreator.loadSave(inputFile);
                    }
                }
                catch(Exception ex)
                {
                    JOptionPane.showMessageDialog(frame,
                            "Error loading File");
                }
                sliderPanel.removeAll();
                //Update the slider panel to fit all the sliders
                sliderPanel.setPreferredSize(new Dimension(Resolutions.sliderPanel[0],(100 * SaveCreator.projectAngles.length)));

                AdditionalOptions.updateSorting();
                currentGroupingLine = 0;
                AdditionalOptions.updateDomains();
                openRanges = true;
                createSortingPanel();

            //Update the field angles, then redraw UI
                for(int i = 0; i < Main.userInputData.get(0).fieldLength; i++)
                {
                    sliderPanel.setLayout(new GridLayout(Main.userInputData.get(0).fields.length, 0));
                    String fieldName = Main.userInputData.get(0).getFieldName(i);
                    int fieldAngle = SaveCreator.projectAngles[i];
                    Main.userInputData.get(0).fieldAngles[i] = fieldAngle;
                    SliderWithIntegration.createPanel(fieldName, fieldAngle, i, sliderPanel);
                }
                drawGraphs();
            repaint();
            revalidate();

            }
        });

        JMenuItem menuItem2 = new JMenuItem("Save Project",
                KeyEvent.VK_T);
        menuItem2.setAccelerator(KeyStroke.getKeyStroke(
                KeyEvent.VK_1, ActionEvent.ALT_MASK));
        menuItem2.getAccessibleContext().setAccessibleDescription(
                "");
        menu.add(menuItem2);

        menuItem2.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                FileNameExtensionFilter filter = new FileNameExtensionFilter("datv", "datv");
                JFileChooser fc = new JFileChooser();
                fc.setFileFilter(filter);
                if(SaveCreator.currentProject == null)
                {
                    //Open the file chooser dialog and get the users file location
                    int returnVal = fc.showSaveDialog(DV.this);
                    //Check if the return value is a valid path
                    try {
                        if (returnVal == JFileChooser.APPROVE_OPTION) {
                            //get the file that the user selected
                            File inputFile = new File(fc.getSelectedFile().getPath() + ".datv");
                            SaveCreator.saveAs(inputFile);
                        }
                    }
                    catch(Exception ex)
                    {
                        JOptionPane.showMessageDialog(frame,
                                "Error Saving File");
                    }
                }
                else
                {
                    try {
                        SaveCreator.save();
                    }
                    catch(Exception E)
                    {
                        JOptionPane.showMessageDialog(frame,
                                "Error Saving File");
                    }
                    }
            }
        });

        JMenuItem menuItem3 = new JMenuItem("Save Project As",
                KeyEvent.VK_T);
        menuItem3.setAccelerator(KeyStroke.getKeyStroke(
                KeyEvent.VK_1, ActionEvent.ALT_MASK));
        menuItem3.getAccessibleContext().setAccessibleDescription(
                "");
        menu.add(menuItem3);

        menuItem3.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                FileNameExtensionFilter filter = new FileNameExtensionFilter("datv", "datv");
                JFileChooser fc = new JFileChooser();
                fc.setFileFilter(filter);
                //Open the file chooser dialog and get the users file location
                int returnVal = fc.showSaveDialog(DV.this);
                    //Check if the return value is a valid path
                try {
                    if (returnVal == JFileChooser.APPROVE_OPTION) {
                        //get the file that the user selected
                        File inputFile = fc.getSelectedFile();
                        SaveCreator.saveAs(inputFile);
                    }
                }
                catch(Exception ex)
                {
                    JOptionPane.showMessageDialog(frame,
                            "Error Saving File");
                }
                }
            });

        JMenuItem menuItem4 = new JMenuItem("Import Data",
                KeyEvent.VK_T);
        menuItem4.setAccelerator(KeyStroke.getKeyStroke(
                KeyEvent.VK_1, ActionEvent.ALT_MASK));
        menuItem4.getAccessibleContext().setAccessibleDescription(
                "");
        menu.add(menuItem4);

        menuItem4.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                loadNewRanges = true;
                //Open the file chooser dialog and get the users file location
                //if(Main.hasClass)
                {
                    //JOptionPane.showMessageDialog(frame,
                     //       "Multiple files are not supported when classes are enabled",
                       //     "Error",
                         //   JOptionPane.ERROR_MESSAGE);
                //}
               // else {
                    FileNameExtensionFilter filter = new FileNameExtensionFilter("csv", "csv");
                    JFileChooser fc = new JFileChooser();
                    fc.setFileFilter(filter);
                    int returnVal = fc.showOpenDialog(DV.this);

                    //Check if the return value is a valid path
                    if (returnVal == JFileChooser.APPROVE_OPTION) {
                        //get the file that the user selected
                        File inputFile = fc.getSelectedFile();
                        //Create a new parser and parse the file
                        DVParser inputParser = new DVParser();
                        try {
                            inputParser.parseData(inputFile);
                        }
                        catch(Exception exc)
                        {
                            JOptionPane.showConfirmDialog(
                                    frame,
                                    "Error opening target file.",
                                    "Error",
                                    JOptionPane.YES_OPTION);
                        }
                        //Update the slider panel to fit all the sliders
                    //Set initial grouping line conditions.
                    Main.sortingLinesPercent = new double[Main.userInputData.size() - 1];
                    Main.sortingLinesValue = new double[Main.userInputData.size() - 1];
                    for(int i = 0; i < Main.sortingLinesPercent.length; i++)
                    {
                        Main.sortingLinesPercent[i] = i;
                    }

                    if(Main.userInputData.size() > 1)
                    {
                   // domainSlider2.setValue((int)Main.sortingLinesPercent[currentGroupingLine]);
                    }
                    }
                    loadNewRanges = true;
                    Range.ranges.clear();
                    createSortingPanel();
                    AdditionalOptions.updateSorting();
                    currentGroupingLine = 0;
                    //Draw the graph
                    AdditionalOptions.updateDomains();
                    drawGraphs();
                    //Update the field angles, then redraw UI
                    updateFieldAngles(sliderPanel);
                    repaint();
                    revalidate();
                }
            }
        });
        
        //Build second menu in the menu bar.
        menu = new JMenu("Edit");
        menu.setMnemonic(KeyEvent.VK_A);
        menu.getAccessibleContext().setAccessibleDescription(
                "");
        menuBar.add(menu);


        menuItem = new JMenuItem("Additional Options",
                KeyEvent.VK_T);
        menuItem.setAccelerator(KeyStroke.getKeyStroke(
                KeyEvent.VK_1, ActionEvent.ALT_MASK));
        menuItem.getAccessibleContext().setAccessibleDescription(
                "");
        menu.add(menuItem);

        menuItem.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                //Open the options menu
                OptionsMenu menu = new OptionsMenu();
            }
        });

        menuItem0 = new JMenuItem("Randomize Angles.",
                KeyEvent.VK_T);
        menuItem0.setAccelerator(KeyStroke.getKeyStroke(
                KeyEvent.VK_1, ActionEvent.ALT_MASK));
        menuItem0.getAccessibleContext().setAccessibleDescription(
                "");
        menu.add(menuItem0);

        menuItem0.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                Random rand = new Random(System.currentTimeMillis());
                sliderPanel.removeAll();
                int count = 0;
                boolean foundBetter = false;
                initialAngles = Arrays.copyOf(Main.userInputData.get(0).fieldAngles, Main.userInputData.get(0).fieldLength);
                double initialAccuracy = Main.accuracy;
                while(count < 1000) {
                    sliderPanel.removeAll();
                    for (int i = 0; i < Main.userInputData.get(0).fieldLength; i++) {
                        sliderPanel.setLayout(new GridLayout(Main.userInputData.get(0).fields.length, 0));
                        String fieldName = Main.userInputData.get(0).getFieldName(i);
                        int fieldAngle = rand.nextInt(181);
                        Main.userInputData.get(0).fieldAngles[i] = fieldAngle;
                        SliderWithIntegration.createPanel(fieldName, fieldAngle, i, sliderPanel);
                    }
                    drawGraphs();
                    count++;
                    if(Main.accuracy > initialAccuracy)
                    {
                        count = 1001;
                        foundBetter = true;
                    }
                }

                if(foundBetter == false)
                {
                    JOptionPane.showMessageDialog(
                            frame,
                            "Was unable to optimize angles.\n",
                            "Error",
                            JOptionPane.OK_OPTION);
                    sliderPanel.removeAll();
                    for (int i = 0; i < Main.userInputData.get(0).fieldLength; i++) {
                        sliderPanel.setLayout(new GridLayout(Main.userInputData.get(0).fields.length, 0));
                        String fieldName = Main.userInputData.get(0).getFieldName(i);
                        int fieldAngle = (int)initialAngles[i];
                        Main.userInputData.get(0).fieldAngles[i] = fieldAngle;
                        SliderWithIntegration.createPanel(fieldName, fieldAngle, i, sliderPanel);
                    }
                }

                drawGraphs();
                repaint();
                revalidate();

                }

            });

        menuItem1 = new JMenuItem("Undo Randomize.",
                KeyEvent.VK_T);
        menuItem1.setAccelerator(KeyStroke.getKeyStroke(
                KeyEvent.VK_1, ActionEvent.ALT_MASK));
        menuItem1.getAccessibleContext().setAccessibleDescription(
                "");
        menu.add(menuItem1);

        menuItem1.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                sliderPanel.removeAll();
                for (int i = 0; i < Main.userInputData.get(0).fieldLength; i++) {
                    sliderPanel.setLayout(new GridLayout(Main.userInputData.get(0).fields.length, 0));
                    String fieldName = Main.userInputData.get(0).getFieldName(i);
                    int fieldAngle = (int)initialAngles[i];
                    Main.userInputData.get(0).fieldAngles[i] = fieldAngle;
                    SliderWithIntegration.createPanel(fieldName, fieldAngle, i, sliderPanel);

            }

            drawGraphs();
            repaint();
            revalidate();
            }
        });

        //Build third menu in the menu bar.
        menu = new JMenu("Help");
        menu.setMnemonic(KeyEvent.VK_A);
        menu.getAccessibleContext().setAccessibleDescription(
                "");
        menuBar.add(menu);

        menuItem = new JMenuItem("User Manual",
                KeyEvent.VK_T);
        menuItem.setAccelerator(KeyStroke.getKeyStroke(
                KeyEvent.VK_1, ActionEvent.ALT_MASK));
        menuItem.getAccessibleContext().setAccessibleDescription(
                "");
        menuItem.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                try {
                    File file = new File(System.getProperty("user.dir") + "\\DVHelp.pdf");
                    Desktop desktop = Desktop.getDesktop();
                    desktop.open(file);
                }
                catch(Exception ex)
                {
                    JOptionPane.showMessageDialog(
                            frame,
                            "Error opening help file.\n" + ex,
                            "Error",
                            JOptionPane.OK_OPTION);
                }
            }
        });
        menu.add(menuItem);

    }

    /**
     * Redraws all graphs
     */
    void drawGraphs() 
    {
        //Set initial boolean state
        int setSize = Main.userInputData.size();
        Main.isMirrored = new boolean[setSize];

        for(int i = 0; i < setSize; i++)
        {
            Main.isMirrored[i] = false;
        }

        if(setSize == 2)
        {
            Main.isMirrored[1] = true;
        }

        graphPanel.removeAll();

        for(int i = 0; i < Main.userInputData.size(); i++)
        {
        	addGraph(Main.userInputData.get(i), i, Main.isMirrored[i]);
        }
        graphPanel.revalidate();

        if(Main.groupingActive) {
            //sortingLabel.removeAll();
            //updateSortingPanel();
            sortingLabel.revalidate();
            analyticsText.removeAll();
            AdditionalOptions.generateConfusionMatrix(analyticsText);
        }

        jp.revalidate();
    }

    /**
     * Draws a single graph and adds it to graph panel
     * @param set Data set to be used for graph
     * @param setCount Index of target set
     * @param isMirrored Flag to tell renderer to mirror points
     */
    public void addGraph(DataSet set, int setCount, boolean isMirrored)
    {
      //Update points in set
      set.generatePoints();
      
    	//Create the main renderer and the data set to go with it.
    	XYLineAndShapeRenderer lineRenderer = new XYLineAndShapeRenderer(true, false);
    	XYSeriesCollection dataset = new XYSeriesCollection();
    	ArrayList<XYSeries> lines = new ArrayList<XYSeries>();
      
      //create renderer to draw domain lines
        XYLineAndShapeRenderer minRenderer = new XYLineAndShapeRenderer(true, false);
        XYLineAndShapeRenderer maxRenderer = new XYLineAndShapeRenderer(true, false);
      XYLineAndShapeRenderer domainRenderer = new XYLineAndShapeRenderer(true, false);
    	XYSeriesCollection domains = new XYSeriesCollection();
        XYSeriesCollection minSorting = new XYSeriesCollection();
        XYSeriesCollection maxSorting = new XYSeriesCollection();
        XYSeries maxLine = new XYSeries(-1, false, true);
        XYSeries minLine = new XYSeries(-2, false, true);
        if(isMirrored)
        {
            maxLine.add(Main.domainMaxValue, 0);
            maxLine.add(Main.domainMaxValue, -(1));
            minLine.add(Main.domainMinValue, 0);
            minLine.add(Main.domainMinValue, -(1));
        }
        else {
            maxLine.add(Main.domainMaxValue, 0);
            maxLine.add(Main.domainMaxValue, (1));
            minLine.add(Main.domainMinValue, 0);
            minLine.add(Main.domainMinValue, (1));
        }

      for(int i = 0; i < Range.ranges.size(); i++) {
          if(isMirrored)
          {
              XYSeries sortingLine = new XYSeries(i, false, true);
              sortingLine.add(Range.ranges.get(i).getMaxValue(), 0);
              sortingLine.add(Range.ranges.get(i).getMaxValue(), -(1));
              maxSorting.addSeries(sortingLine);

              XYSeries sortingLineTwo = new XYSeries(i, false, true);
              sortingLineTwo.add(Range.ranges.get(i).getMinValue(), 0);
              sortingLineTwo.add(Range.ranges.get(i).getMinValue(), -(1));
              minSorting.addSeries(sortingLineTwo);
          }
          else {
              XYSeries sortingLine = new XYSeries(i, false, true);
              sortingLine.add(Range.ranges.get(i).getMaxValue(), 0);
              sortingLine.add(Range.ranges.get(i).getMaxValue(), (1));
              maxSorting.addSeries(sortingLine);

              XYSeries sortingLineTwo = new XYSeries((i), false, true);
              sortingLineTwo.add(Range.ranges.get(i).getMinValue(), 0);
              sortingLineTwo.add(Range.ranges.get(i).getMinValue(), (1));
              minSorting.addSeries(sortingLineTwo);
          }
      }

      //Add new series to collection
      domains.addSeries(minLine);
      domains.addSeries(maxLine);

      
      //Create Renderer to draw endpoint, sorting, and timeline points
      XYLineAndShapeRenderer endpointRenderer = new XYLineAndShapeRenderer(false, true);
      XYSeriesCollection endpoints = new XYSeriesCollection();
      XYSeries endpointSeries = new XYSeries(0, false, true);
      XYLineAndShapeRenderer timelineRenderer = new XYLineAndShapeRenderer(false, true);
      XYSeriesCollection timeline = new XYSeriesCollection();
      XYSeries timelineSeries = new XYSeries(0, false, true);

    	//Populate the series with values from the dataset
    	for(int i = 0; i < set.members.length; i++)
    	{
    		double endpoint = 0;
    		lines.add(new XYSeries(i, false, true));
         lines.get(i).add(0, 0);
    		for(int j = 0; j < set.fieldLength; j++)
    		{
                if(isMirrored)
                {
                    lines.get(i).add(set.members[i].points[j][0], -set.members[i].points[j][1]);
                }
                else {
                    lines.get(i).add(set.members[i].points[j][0], set.members[i].points[j][1]);
                }
    			if(j == (set.fieldLength - 1))
    			{
               endpoint = set.members[i].points[j][0];
               if(Main.domainActive == false)
		    	      {
                          if(isMirrored)
                          {
                              endpointSeries.add(set.members[i].points[j][0], -set.members[i].points[j][1]);
                              timelineSeries.add(set.members[i].points[j][0], 0);
                          }
                          else {
                              endpointSeries.add(set.members[i].points[j][0], set.members[i].points[j][1]);
                              timelineSeries.add(set.members[i].points[j][0], 0);
                          }

                  }
               else if(endpoint >= Main.domainMinValue && endpoint <= Main.domainMaxValue)
	    		      {
                          if(isMirrored)
                          {
                              endpointSeries.add(set.members[i].points[j][0], -set.members[i].points[j][1]);
                              timelineSeries.add(set.members[i].points[j][0], 0);
                          }
                          else {
                              endpointSeries.add(set.members[i].points[j][0], set.members[i].points[j][1]);
                              timelineSeries.add(set.members[i].points[j][0], 0);
                          }
	    		      }
	    		   else
	    		      {

	    		      }     
    			}
    		}

    		timelineSeries.add(-(Main.userInputData.get(0).fieldLength), 0);
            timelineSeries.add((Main.userInputData.get(0).fieldLength), 0);
         
		    if(Main.domainActive == false)
		    	{
		   		dataset.addSeries(lines.get(i));
		    	}
	 		else if(endpoint >= Main.domainMinValue && endpoint <= Main.domainMaxValue)
	    		{
	    			dataset.addSeries(lines.get(i));
	    		}
	    		else
	    		{

	    		}
    	}
      
      //Add the new made data to the series collection
      endpoints.addSeries(endpointSeries);
      timeline.addSeries(timelineSeries);

    	//Create the chart to be drawn
    	JFreeChart chart = ChartFactory.createXYLineChart(
    		"",
    		" ",
    		" ",
    		dataset,
    		PlotOrientation.VERTICAL,
    		false, true, false);
      
      //Format chart
      chart.setBorderVisible(false);
      chart.setPadding(RectangleInsets.ZERO_INSETS);
      
    	//Get the plot from the chart
    	XYPlot plot = (XYPlot) chart.getPlot();
      
      //Create simple rng for colors
      Random rand = new Random(setCount);
      
      //Format the plot
      plot.setDrawingSupplier(new DefaultDrawingSupplier(
                new Paint[] {new Color(rand.nextInt(201), rand.nextInt(201), rand.nextInt(201))},
                DefaultDrawingSupplier.DEFAULT_OUTLINE_PAINT_SEQUENCE,
                DefaultDrawingSupplier.DEFAULT_STROKE_SEQUENCE,
                DefaultDrawingSupplier.DEFAULT_OUTLINE_STROKE_SEQUENCE,
                DefaultDrawingSupplier.DEFAULT_SHAPE_SEQUENCE));
        plot.getRangeAxis().setVisible(false);
        plot.getDomainAxis().setVisible(false);
        plot.setRangeGridlinesVisible(false);
        plot.setOutlinePaint(null);
        plot.setOutlineVisible(false);
        plot.setInsets(RectangleInsets.ZERO_INSETS);

    	//Set the domain and range of the graph
        if(isMirrored)
        {
            ValueAxis domainView = plot.getDomainAxis();
            //domainView.setRange(-(Main.userInputData.get(0).fieldLength), Main.userInputData.get(0).fieldLength);
            ValueAxis rangeView = plot.getRangeAxis();
            //rangeView.setRange(-Main.userInputData.get(0).fieldLength, 0);
        }
        else {
            ValueAxis domainView = plot.getDomainAxis();
           //domainView.setRange(-(Main.userInputData.get(0).fieldLength), Main.userInputData.get(0).fieldLength);
            ValueAxis rangeView = plot.getRangeAxis();
            //rangeView.setRange(0, Main.userInputData.get(0).fieldLength);
        }

        XYIntervalSeriesCollection bars = new XYIntervalSeriesCollection();
        XYBarRenderer barRenderer = new XYBarRenderer();

        //Create bar chart if enabled
        if(isBars)
        {
            int[] barRanges = new int[100];
            double memberCount = 0;
            for(int i = 0; i < Main.userInputData.size(); i++)
            {
                memberCount += Main.userInputData.get(i).members.length;
            }
                //Count the number of points in each range
                for(int j = 0; j < set.members.length; j++)
                {
                    double endpoint = set.members[j].points[set.fieldLength - 1][0];
                    int rangeCount = -1;
                    double sizeCount = - set.fieldLength;
                    while(endpoint > sizeCount)
                    {
                        rangeCount++;
                        sizeCount += (Main.userInputData.get(0).length() * 2) / 100.0;
                    }
                    barRanges[rangeCount]++;
                }
            double interval = (set.length() * 2) / 100.0;
            double tempMaxInterval = - set.fieldLength;;
            double oldInterval = - set.fieldLength;
            //Add series to data collection
            for(int i = 0; i < 100; i++)
            {
                if(isMirrored) {
                    tempMaxInterval += interval;
                    XYIntervalSeries bar = new XYIntervalSeries(i, false, true);
                    bar.add(interval, oldInterval, tempMaxInterval, -barRanges[i] / memberCount, -1, 0);
                    bars.addSeries(bar);
                    oldInterval = tempMaxInterval;
                }
                else
                {
                    tempMaxInterval += interval;
                    XYIntervalSeries bar = new XYIntervalSeries(i, false, true);
                    bar.add(interval, oldInterval, tempMaxInterval, barRanges[i] / memberCount, 0, 1);
                    bars.addSeries(bar);
                    oldInterval = tempMaxInterval;
                }
            }
        }

    	//Set the renderers and their datasets
    	plot.setRenderer(0, lineRenderer);
    	plot.setDataset(0, dataset);
      
      plot.setRenderer(1, domainRenderer);
    	plot.setDataset(1, domains);
      
      plot.setRenderer(2, endpointRenderer);
    	plot.setDataset(2, endpoints);

    	if(isBars) {
            plot.setRenderer(3, barRenderer);
            plot.setDataset(3, bars);
        }
        else
        {
            plot.setRenderer(3, timelineRenderer);
            plot.setDataset(3, timeline);
        }

    	plot.setRenderer(4, minRenderer);
        plot.setDataset(4, minSorting);

        plot.setRenderer(5, maxRenderer);
        plot.setDataset(5, maxSorting);

    	//Set renderer colors
        for(int i = 0; i < Main.userInputData.size(); i++) {
            if(i == currentGroupingLine)
            {
                plot.getRendererForDataset(plot.getDataset(4)).setSeriesPaint(i, activeThresholdLine);
                plot.getRendererForDataset(plot.getDataset(5)).setSeriesPaint(i, activeThresholdLine);
            }
            else
            {
                plot.getRendererForDataset(plot.getDataset(4)).setSeriesPaint(i, inactiveThresholdLine);
                plot.getRendererForDataset(plot.getDataset(5)).setSeriesPaint(i, inactiveThresholdLine);
            }
        }
        //Create the graph panel and add it to the main panel
    	ChartPanel tempPanel = new ChartPanel(chart);
        tempPanel.setMouseWheelEnabled(true);
        tempPanel.setPreferredSize(new Dimension(Resolutions.tempPanel[0], Resolutions.tempPanel[1]));
    	graphPanel.add(tempPanel);
    }
}
