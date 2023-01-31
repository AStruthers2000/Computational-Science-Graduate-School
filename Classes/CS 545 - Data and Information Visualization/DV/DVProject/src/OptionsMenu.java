
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
/**
*Create a void method that, when called, opens a new JFrame.
*JFrame should be standard windows style interface and should have options to toggle on domain lines and to toggle on grouping line
*Options should toggle booleans 'domainActive' and 'groupingActive' respectively
*Above booleans are found in main class.
**/
public class OptionsMenu extends JPanel implements ItemListener
{

    //Create the check boxes.
    JCheckBox domainActiveButton;
    JCheckBox groupingActiveButton;

    /**
     * Opens a basic option menu for the user to interact with.
     */
	public OptionsMenu()
	{
      super(new BorderLayout());
      
      //Create and set up the window.
        JFrame frame = new JFrame("Additional Options");
        frame.setResizable(false);
        //frame.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);

        //Create and set up the content pane.
        //JComponent newContentPane = new CheckBoxDemo();
        //newContentPane.setOpaque(true); //content panes must be opaque
       // frame.setContentPane(newContentPane);
    
    //Register a listener for the check boxes.
    
    //first checkbox  
    domainActiveButton = new JCheckBox("domainActive");
    domainActiveButton.setMnemonic(KeyEvent.VK_C); 
    domainActiveButton.setSelected(Main.domainActive);
    domainActiveButton.setToolTipText("Toggle range restricted viewing on or off.");
    
    //second checkbox
    groupingActiveButton = new JCheckBox("groupingActive");
    groupingActiveButton.setMnemonic(KeyEvent.VK_G); 
    groupingActiveButton.setSelected(Main.groupingActive);

    //Register a listener for the check boxes.
    domainActiveButton.addActionListener(new ActionListener()
    {
      public void actionPerformed(ActionEvent e) {
         Main.domainActive = !Main.domainActive;
      }
    });
    groupingActiveButton.addItemListener(this);
    
    //Put the check boxes in a column in a panel
        JPanel checkPanel = new JPanel();
        checkPanel.add(domainActiveButton);
        //checkPanel.add(groupingActiveButton);
        
        add(checkPanel, BorderLayout.LINE_START);
       //frame.add(pictureLabel, BorderLayout.CENTER);
        //frame.setBorder(BorderFactory.createEmptyBorder(20,20,20,20));

        JPanel colors = new JPanel();
        JButton activeButton = new JButton("Set active sorting line color");
        activeButton.setToolTipText("Set color of active threshold lines.");
        activeButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                Color newColor = JColorChooser.showDialog(
                        frame,
                        "Choose Line Color",
                        frame.getBackground());
                if(newColor != null){
                    DV.activeThresholdLine = newColor;
                    Main.dv.drawGraphs();
                }
            }
        });

        colors.add(activeButton);

        JButton inactiveButton = new JButton("Set inactive sorting line color");
        inactiveButton.setToolTipText("Set color of inactive threshold lines.");
        inactiveButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                Color newColor = JColorChooser.showDialog(
                        frame,
                        "Choose Line Color",
                        frame.getBackground());
                if(newColor != null){
                    DV.inactiveThresholdLine = newColor;
                    Main.dv.drawGraphs();
                }
            }
        });

        colors.add(inactiveButton);
        ///colors.setLayout(new BoxLayout(colors, BoxLayout.Y_AXIS));

        JPanel mainPanel = new JPanel();
        //mainPanel.setLayout(new BoxLayout(mainPanel, BoxLayout.Y_AXIS));
        mainPanel.add(checkPanel);
        mainPanel.add(colors);

        frame.add(mainPanel);
        //Display the window.
        frame.pack();
        frame.setVisible(true);
		
	}
   /** Listens to the check boxes. */
    public void itemStateChanged(ItemEvent e) {
        int index = 0;
        //DV dv = new Dv();
        Object source = e.getItemSelectable();
        
                //Now that we know which button was pushed, find out
        //whether it was selected or deselected.
        if (e.getStateChange() == ItemEvent.DESELECTED) {
           
           if (source == domainActiveButton) {
               index = 0;
              Main.domainActive =false;
           } else if (source == groupingActiveButton) {
               index = 1;
              Main.groupingActive = false;
           }
        }

        if (source == domainActiveButton) {
            index = 0;
           Main.domainActive = true;
        } else if (source == groupingActiveButton) {
            index = 1;
           Main.groupingActive = true;
        }

    }
}