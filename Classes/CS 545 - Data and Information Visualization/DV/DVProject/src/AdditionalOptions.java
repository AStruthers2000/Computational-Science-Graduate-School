
import javax.swing.*;


public class AdditionalOptions
{

	/**
	*A void method that converts the percent values found in DomainMinPercent and DomainMaxPercent
	*into their corresponding x values.
	*
	*the graph is x units long, where x is the number of fields
	*
	*Store the generated x values in DomainMinValue and DomainMaxValue
	*
	*All variables to be changed are found in Main.java
	**/
	public static void updateDomains()
	{
            Main.domainMinValue = (((2 * Main.domainMinPercent) * Main.userInputData.get(0).fieldLength) /100) - Main.userInputData.get(0).fieldLength;
            Main.domainMaxValue = (((2 * Main.domainMaxPercent) * Main.userInputData.get(0).fieldLength) /100) - Main.userInputData.get(0).fieldLength;
	}

    /**
	*A void method that converts the percent values found in sortingLinesPercent[]
	*into their corresponding x values.
	*
	*the graph is x units long, where x is the number of fields
	*
	*Store the generated x values in SortingLinesValue
	*
	*The method should then iterate over each DataObject in each DataSet and assess each line endpoint. 
	*
	*It should then record the number of values in each set in each range in sortingDistribution[set][inRange].
	*
	*sortingDistribution should be resized to be #fields x #fields
	*
	*All variables to be changed are found in Main.java
	**/
	public static void updateSorting()
	{
	}

	/**
	*A void method that uses sortingDistribution[set][inRange] to generate a confusion matrix
	*
	*The confusion matrix is based on perevious research
	*
	*Write the confusion matrix to the input JTextArea
     *
     * @param writeArea The JTextArea which the method writes to.
	**/
	public static void generateConfusionMatrix(JTextArea writeArea) {
            boolean validThresholds;
        if (Range.validRange()) validThresholds = true;
        else validThresholds = false;
        for (int i = 0; i < Range.ranges.size(); i++) {
                if (Range.validRange()) {

                } else {
                    // validThresholds = false;
                }
            }
            if (validThresholds == true) {
                double[][] sortingLinesDistribution = new double[Range.ranges.size()][Range.ranges.size()];
                for (int i = 0; i < Main.userInputData.size(); i++) {
                    for (int j = 0; j < Main.userInputData.get(i).members.length; j++) {
                        double endpoint = Main.userInputData.get(i).members[j].points[Main.userInputData.get(0).fieldLength - 1][0];
                        for (int k = 0; k < Range.ranges.size(); k++) {
                            if (Range.ranges.get(k).inRange(endpoint)) {
                                sortingLinesDistribution[i][k]++;
                            }
                        }
                    }
                }
                int total = 0;
                double inTarget = 0;
                String header = "Real\t\tPredicted Class\nClass\t";
                StringBuilder sb = new StringBuilder();
                sb.append(header);

                for (int i = 0; i < Range.ranges.size(); i++) {
                    sb.append(i + 1 + "\t  ");
                }

                //Generate total value
                for (int i = 0; i < Main.userInputData.size(); i++) {
                    total += Main.userInputData.get(i).members.length;
                }

                //Make header
                for (int i = 0; i < sortingLinesDistribution.length; i++) {
                    //Class i
                    sb.append("\n" + (i + 1) + "\t");
                    for (int j = 0; j < sortingLinesDistribution[i].length; j++) {
                        sb.append(sortingLinesDistribution[i][j] + "\t");
                        if (i == j) {
                            inTarget += sortingLinesDistribution[j][i];
                        }
                    }
                }

                double accuracyNum;
                accuracyNum = inTarget / total * 100;
                Main.accuracy = accuracyNum;
                String accuracy = "\nAccuracy is: " + accuracyNum + "%\n";
                sb.append(accuracy);
                String table = sb.toString();
                writeArea.setText(table);
            } else {
                writeArea.setText("INVALID THRESHOLDS, PLEASE REMOVE OVERLAP.");
            }
        }

}