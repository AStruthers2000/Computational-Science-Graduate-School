import java.util.Arrays;

public class DataSet{                            
        String name;                                            //name of the DataSet
        String[] fields;                                        //An array containing the name of each data field
        double[] fieldAngles;                                   //an array containing the angle that each data field should be rendered at
        DataObject[] members;                                   //an array containing each DataObject in the DataSet
        int fieldLength;                                        //the number of fields in the DataSet

    /**
     * DataSet Constructoe
     * @param name name of dataset
     * @param fieldNames Array of name of the dataset's fields
     * @param objects An array of data objects in the data set
     */
    public DataSet(String name, String[] fieldNames, DataObject[] objects){ //creates a new DataSet using the given input
            this.name = name;
            fields = fieldNames;
            members = objects;
            fieldLength = fields.length;
            fieldAngles = new double[fieldLength];
            for(int i = 0; i < fieldLength; i++)
            {
                fieldAngles[i] = 45;
            }
        }

        public void generatePoints()
        {
            for(int i = 0; i < members.length; i++)
            {
                members[i].updatePoints(Main.userInputData.get(0).fieldAngles);
            }
        }

    /**
     * Return field name at position n
     * @param position position n
     * @return The field name
     */
    public String getFieldName(int position){
            return fields[position];
        }

    /**
     * returns dataobject at position n
     * @param position position n
     * @return Data object at position m
     */
    public DataObject getMember(int position){
            return members[position];
        }

    /**
     *
     * @return Number of fields
     */
    public int length(){
            return fieldLength; 
        }

    /**
     * Gets field angle at position n
     * @param position n
     * @return Field angle at position n
     */
    public double getFieldAngle(int position){
            return fieldAngles[position];
        }

    /**
     * Sets current field angles to target
     * @param userFieldAngles Target field angles
     */
    public void updateFieldAngles(double[] userFieldAngles){
            fieldAngles = userFieldAngles;
        }
    
    }