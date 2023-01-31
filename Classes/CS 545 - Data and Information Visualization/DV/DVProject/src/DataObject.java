import java.util.Arrays;

import static java.lang.Math.cos;

public class DataObject{                                        //DataObject class with 2 variables, index and value
        int index;                                              //index: the objects position in the dataset
        double[] values;                                        //values: the array of normalized values for this DataObject
        double[][] points;

    /**
     * COnstructor for DataObject
     * @param val An array of normalized values
     * @param index Which data object this is
     */
    public DataObject(double[] val, int index){             //Creating the DataObject with variables val and index
            values = val;                                       //assigns val to values
            this.index = index;                                        //assigns ind to index
            
        }
        
        public double getValue(int position){                   //getValue: returns value at a specified position 
            return values[position];
        }
        
        public String toString(String[] fields){                //toString: returns a string of all values, each seperated by a comma and space
            return (Arrays.toString(values));
        }

    /**
     * Updates the endpoints for use in lines
     * @param angles list of angles to draw lines to
     */
    public void updatePoints(double[] angles)
    {
        points = new double[values.length][2];
        int counter = 0;
        double[] temp = getCoords(values[counter], angles[counter]);
        points[counter][0] = temp[0];
        points[counter][1] = temp[1];
        if(values.length > counter)
        {
            updatePoints(temp[0], temp[1], (counter + 1), angles);
        }
        else
        {}
    }

    /**
     * Updates the endpoints for use in lines
     * @param prevX the x value of the previous line
     * @param prevY the y value of the previous live
     * @param counter which line it currently is
     * @param angles list of angles to draw lines to
     */
   private void updatePoints(double prevX, double prevY, int counter, double[] angles)
    {
        double[] temp = getCoords(values[counter], angles[counter]);
        points[counter][0] = (temp[0] + prevX);
        points[counter][1] = (temp[1] + prevY);
        counter++;
        if(values.length > counter)
        {
            updatePoints(points[counter - 1][0], points[counter - 1][1], counter , angles);
        }
        else
        {}
    }

    /**
     * Generate and enpoint from an angle and a length
     * @param length Desired line length
     * @param angle desired angle
     * @return Coordinate points of end of line
     */
    private double[] getCoords(double length, double angle)
    {
        double[] output = new double[2];
        if(angle > 88 && angle < 90)
        {
            //angle = 88;
        }
        else if(angle >= 90 && angle < 92)
        {
            //angle = 92;
        }
         
        if(angle > 90)
        {
         angle = 180 - angle;
         output[0] = -((Math.cos(Math.toRadians(angle))) * length);
         output[1] = (Math.sin(Math.toRadians(angle))) * length;
        }
        else
        {
         output[0] = (Math.cos(Math.toRadians(angle))) * length;
         output[1] = (Math.sin(Math.toRadians(angle))) * length;
        }
        return output;
    }
}