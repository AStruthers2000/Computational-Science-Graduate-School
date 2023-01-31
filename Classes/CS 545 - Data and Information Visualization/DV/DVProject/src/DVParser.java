import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files; 
import java.nio.file.Path; 
import java.nio.file.Paths; 
import java.util.*;
import java.lang.Math.*;
import java.util.Scanner;
import java.util.ArrayList;

class DVParser {
    //Values for normalization
    static boolean isFirst = true;
    public static ArrayList<Double> minValues = new ArrayList();
    public static ArrayList<Double> maxValues = new ArrayList();
    public static ArrayList<File> files = new ArrayList();
    public static int classNumber;

    /**
     * Default constructor
     */
    public DVParser() {
    }

    /**
     * Parses csv to be rendered and stores in userInputData
     * @param loc target file
     * @throws IOException
     */
    public void parseData(File loc) throws IOException {
        if (Main.hasClass && Main.needSplitting) {
            addClasses(loc);
        } else {
            files.add(loc);
        }

        for (int i = 0; i < files.size(); i++) {
            //Import .csv file into java
            String[][] rawInputNext = this.getStringFromCSV(files.get(i));
            //Check if column 0 is the index. If it is, remove it.
            if (Main.hasIndex) {
                rawInputNext = purgeIndex(rawInputNext);
            }
            if (Main.hasClass) {
                rawInputNext = purgeClasses(rawInputNext);
            }
            //Transform raw input into double array
            double[][] inputValuesNext = this.stringToValues(rawInputNext);
            //Normalize values
            double[][] normalizedValuesNext = this.normalize(inputValuesNext);
        }

        Main.userInputData.clear();

        //Do it all again to get correct normalization
        for (int i = 0; i < files.size(); i++) {
            //Import .csv file into java
            String[][] rawInputNext = this.getStringFromCSV(files.get(i));
            //Purge index if needed
            if (Main.hasIndex) {
                rawInputNext = purgeIndex(rawInputNext);
            }
            if (Main.hasClass) {
                rawInputNext = purgeClasses(rawInputNext);
            }
            //Transform raw input into field names array for DataSet object
            String[] inputFieldNamesNext = this.stringToFieldNames(rawInputNext);
            //Transform raw input into double array
            double[][] inputValuesNext = this.stringToValues(rawInputNext);
            //Normalize values
            double[][] normalizedValuesNext = this.normalize(inputValuesNext);
            //Transform values in to DataObjects
            DataObject[] inputObjectsNext = this.valueToDataObjects(normalizedValuesNext);
            //Create the DataSet and return it
            DataSet output = this.createDataSet(inputFieldNamesNext, inputObjectsNext);
            //Add to user input data
            Main.userInputData.add(output);
        }
    }

    /**
     * Splits a multi class file into classes
     * @param target target file
     * @throws IOException
     */
    private void addClasses(File target) throws IOException {
        //System.out.println(temp[0][temp[0].length - 1]);
        String[][] temp = getStringFromCSV(target);
        HashSet<String> classesFound = new HashSet();
        for (int i = 1; i < temp.length; i++) {
            classesFound.add(temp[i][temp[0].length - 1]);
        }
        Iterator<String> it = classesFound.iterator();
        DVParser.classNumber = classesFound.size();

        while (it.hasNext()) {
            BufferedReader br = new BufferedReader(new FileReader(target));
            String tempFields = br.readLine();

            String currentClass = it.next();

            //Create new file and its writes
            File tempFile = new File(("projectFiles\\DV" + currentClass + target.getName()));
            FileWriter fw = new FileWriter(tempFile);
            BufferedWriter bw = new BufferedWriter(fw);
            bw.write(tempFields + "\n");

            for (int i = 1; i < temp.length; i++) {
                tempFields = br.readLine();
                if (currentClass.equals(temp[i][temp[0].length - 1])) {
                    if (tempFields != null) {
                        //Write field names to file
                        bw.write(tempFields + "\n");
                    }
                }
            }
            bw.close();
            files.add(tempFile);
            br.close();
        }

    }

    /**
     * Takes string path to .csv file and returns Strign array of input values.
     * @param location target file
     * @return unparsed input array
     */
	private String[][] getStringFromCSV(File location)
	{
      
		//Create a buffered reader object, using try to make sure file is handled correctly
		try (Scanner reader = new Scanner(location))
		{
			//Create a String arraylist to store split values in
			List<String> unsplitInput = new ArrayList<String>();
			String line;

			//Loop through, reading each line and adding it to the String array
			do
			{
				//Read next line of .csv
				line = reader.nextLine();

				//Add string to arraylist
				unsplitInput.add(line);
			}while(reader.hasNextLine());

			//Convert String array list into 2d String array, then return it
			String[][] output = new String[unsplitInput.size()][0];

			for(int i = 0; i < unsplitInput.size(); i++)
			{
				output[i] = unsplitInput.get(i).split(",");
			}
         	return output;
		}
      catch(IOException e)
      {}
      return null;
	}

    /**
     * Removes index column form input array
     * @param input unparsed input array
     * @return pruned input array
     */
	private String[][] purgeIndex(String[][] input)
    {
        String[][] output = new String[input.length][input[0].length - 1];

        for(int i = 0; i < input.length; i++)
        {
            if(input[i][0] != null) {
                for (int j = 0; j < input[0].length - 1; j++) {
                    output[i][j] = input[i][j + 1];
                }
            }
            else
            {

            }
        }

        return output;
    }

    /**
     * Removes class column from input array
     * @param input unparsed input array
     * @return pruned inpuit array
     */
    private String[][] purgeClasses(String[][] input)
    {
        String[][] output = new String[input.length][input[0].length - 1];

        for(int i = 0; i < input.length; i++)
        {
            for(int j = 0; j < input[0].length - 1; j++)
            {
                output[i][j] = input[i][j];
            }
        }

        return output;
    }

    /**
     * Gets array of field names from unparsed input array
     * @param inputRaw unparsed input array
     * @return An array of field names
     */
	private String[] stringToFieldNames(String[][] inputRaw)
	{
		//Create a new String array that the output will be stored in
		String[] output = new String[inputRaw[0].length];

		//Loop the through the first row of the string array, setting the field name values to the values in the first row
		for(int i = 0; i < inputRaw[0].length; i++)
		{
			output[i] = inputRaw[0][i];
		}

		return output;

	}

    /**
     * Converts unparsed input array to an array of values
     * @param inputRaw
     * @return An array of values
     */
	private double[][] stringToValues(String[][] inputRaw)
	{
      //Establsih height and width as variables
      int height = inputRaw.length - 1;
      int width = inputRaw[0].length;
      //Create new array using the height and width values
      double[][] output = new double[height][width];
      //Starting at the second row, iterate through array of values, moving them all into the new array
      for(int i = 0; i < height; i++)
      {

         for(int j = 0; j < width; j++)
         {
            //Turn the Strings into doubles and store in new array
            output[i][j] = Double.parseDouble(inputRaw[i+1][j]);
         }
      }
      //Return output
      return output;
	}

	private void setExtremes(double[][] values)
    {

    }

    /**
     * Normalizes values array
     * @param values parsed values
     * @return An array of parsed, normalized values
     */
	private double[][] normalize(double[][] values)
	{
      //Create a for loop iterating through each array
      for(int i = 0; i < values[0].length; i++) {
          isFirst = true;
          for (int j = 0; j < values.length; j++) {
              double tempValue = values[j][i];
              if (isFirst) {
                  maxValues.add(tempValue);
                  minValues.add(tempValue);
                  isFirst = false;
              } else {
                  if (tempValue > maxValues.get(i)) {
                      maxValues.set(i, tempValue);
                  } else if (tempValue < minValues.get(i)) {
                      minValues.set(i, tempValue);
                  }
              }
          }
      }

         //Iterate over each value, normalizing them to between 1 and 0.
         for(int j = 0; j < values[0].length; j++)
         {
             double tempMin = minValues.get(j);
             double tempMax = maxValues.get(j);
             for(int i = 0; i < values.length; i++) {
                 values[i][j] = ((values[i][j] - tempMin) / ((tempMax - tempMin))) + .1;
             }
         }
      isFirst = false;
      return values;
	}

    /**
     * converts array of values to an array of data objects
     * @param values Array of parsed, normalized values
     * @return Output array of data objects
     */
	private DataObject[] valueToDataObjects(double[][] values)
	{
        //Create an array of dtat objects to store output
        DataObject[] output = new DataObject[values.length];
        //use a for loop to turn each array into a data object.
        for(int i = 0; i < values.length; i++)
        {
            output[i] = new DataObject(values[i], i);
        }
        //Return output
        return output;
	}

    /**
     * Creates a new data set
     * @param fieldNames arrray of field names
     * @param objects array of data objects
     * @return Output data set
     */
	private DataSet createDataSet(String[] fieldNames, DataObject[] objects)
	{
        //return a newly created DataSet
        return new DataSet("temp", fieldNames, objects);
	}
}