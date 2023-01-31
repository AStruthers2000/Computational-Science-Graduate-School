import java.io.*;
import java.util.StringTokenizer;

public class SaveCreator {

    static File currentProject;
    static int[] projectAngles;
    static double[] projectSortingLine;

    /**
     * Save file at target location
     * @param loc Target location
     * @throws IOException
     */
    public static void saveAs(File loc) throws IOException
    {
        currentProject = loc;
        //Create file write
        FileWriter fw = new FileWriter(loc);
        BufferedWriter saveWriter = new BufferedWriter(fw);

        //Write if it has index
        if(Main.hasIndex)
        {
            saveWriter.write("1\n");
        }
        else
        {
            saveWriter.write("0\n");
        }
        //Write if it has classes
        if(Main.hasClass)
        {
            saveWriter.write("1\n");
        }
        else
        {
            saveWriter.write("0\n");
        }
        //Write file list to target
        for(int i = 0; i < DVParser.files.size(); i++)
        {
            saveWriter.write(DVParser.files.get(i).getAbsolutePath() + ",");
        }
        saveWriter.write("\n");
        //Write angle list to to target
        for(int i = 0; i < Main.userInputData.get(0).fieldLength; i++)
        {
            saveWriter.write(Main.userInputData.get(0).fieldAngles[i] + ",");
        }
        saveWriter.write("\n");

        //Write angle list to to target
        for(int i = 0; i < Range.ranges.size(); i++)
        {
            saveWriter.write(Range.ranges.get(i).maximumPercent + ",");
            saveWriter.write(Range.ranges.get(i).minimumPercent + ",");
        }
        saveWriter.write("\n");

        saveWriter.close();
    }

    /**
     * Saves with project name, otherwise forces save as
     * @throws IOException
     */
    public static void save() throws IOException
    {
        saveAs(currentProject);
    }

    /**
     * loads a save from loc
     * @param loc File to be opened
     * @throws IOException
     */
    public static void loadSave(File loc) throws IOException
    {
        Main.needSplitting = false;
        //Create readers
        FileReader fr = new FileReader(loc);
        BufferedReader saveReader = new BufferedReader(fr);
        //Read if it has index
        String indexChar = saveReader.readLine();
        if(indexChar.equals("1"))
        {
            Main.hasIndex = true;
        }
        String classChar = saveReader.readLine();
        if(classChar.equals("1"))
        {
            Main.hasClass = true;
        }

        //Tokenize strings
        String filePaths = saveReader.readLine();
        StringTokenizer saveSplitter = new StringTokenizer(filePaths, ",", false);
        //Parse all files.
        DVParser saveParser = new DVParser();
        while(saveSplitter.hasMoreTokens())
        {
            File temp = new File(saveSplitter.nextToken());
            saveParser.parseData(temp);
        }

        //Load angles
        String angles = saveReader.readLine();
        StringTokenizer angleSplitter = new StringTokenizer(angles, ",", false);
        int i = 0;
        int length = angleSplitter.countTokens();
        projectAngles = new int[length];
        while(angleSplitter.hasMoreTokens())
        {
            double temp = Double.parseDouble(angleSplitter.nextToken());
            projectAngles[i] = (int)temp;
            i++;
        }


        //Load sorting lines
        Main.dv.loadNewRanges = false;
        Main.dv.openRanges = true;
        Main.dv.splittableRanges = saveReader.readLine();

        saveReader.close();
    }
}
