import java.util.ArrayList;

import static java.util.Collections.max;

public class Range {

    //Static array list of all Ranges
    public static ArrayList<Range> ranges = new ArrayList();

    //Minimum and maximum values of the range
    int minimumPercent;
    int maximumPercent;
    //The range's graph index
    int ownerClass;

    /**
     * Range Contrusctor
     * @param min Starting Min
     * @param max Starting Max
     * @param owner index of Range
     */
    Range(int min, int max, int owner)
    {
        //Set initial values
        minimumPercent = min;
        maximumPercent = max;
        ownerClass = owner;

        //add self to arraylist
        ranges.add(this);
    }

    /**
     * Sets range to new values
     * @param newMin new minimum value
     * @param newMax new maximum values
     */
    public void setRange(int newMin, int newMax)
    {
            this.maximumPercent = newMax;
            this.minimumPercent = newMin;
    }

    /**
     * Checks if a given values is in range
     * @param location Target value to check
     * @return If it is in range
     */
    public boolean inRange(double location)
    {
        if(location >= this.getMinValue() && location <= this.getMaxValue())
        {
            return true;
        }
        return false;
    }

    /**
     * Checks if any existing ranges are overlapping
     * @return If any are overlapping
     */
    public static boolean validRange()
    {
        if(ranges.size() == 1)
        {
            return true;
        }
        for(int i = 0; i < ranges.size(); i++)
        {
            for(int j = i + 1; j < ranges.size(); j++)
            {
                double percentMinOne = Range.ranges.get(i).minimumPercent;
                double percentMinTwo = Range.ranges.get(j).minimumPercent;
                double percentMaxOne = Range.ranges.get(i).maximumPercent;
                double percentMaxTwo = Range.ranges.get(j).maximumPercent;

                if(percentMaxOne <= percentMinTwo)
                {

                }
                else if(percentMaxTwo <=  percentMinOne)
                {

                }
                else
                {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * @return Minimum value of range
     */
    public double getMaxValue()
    {
        return (((2 * maximumPercent) * Main.userInputData.get(0).fieldLength) /100.0) - Main.userInputData.get(0).fieldLength;
    }

    /**
     * @return Maximum value of range
     */
    public double getMinValue()
    {
        return (((2 * minimumPercent) * Main.userInputData.get(0).fieldLength) /100.0) - Main.userInputData.get(0).fieldLength;
    }
}
