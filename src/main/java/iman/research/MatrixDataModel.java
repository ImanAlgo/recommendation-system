package iman.research;

import java.io.IOException;

import com.google.common.collect.BiMap;

import net.librec.common.LibrecException;
import net.librec.conf.Configuration;
import net.librec.data.DataModel;
import net.librec.data.model.AbstractDataModel;
import net.librec.math.structure.DataSet;
import net.librec.math.structure.SparseMatrix;

/**
 * A <tt>TextDataModel</tt> represents a data access class to the CSV format
 * input.
 *
 * @author WangYuFeng
 */
public class MatrixDataModel extends AbstractDataModel implements DataModel {

    private SparseMatrix data;
    private SparseMatrix dateTimeData;
    private BiMap<String, Integer> userMappingData;
    private BiMap<String, Integer> itemMappingData;

    public MatrixDataModel(Configuration conf, SparseMatrix data, BiMap<String, Integer> userMappingData, BiMap<String, Integer> itemMappingData){
        this.conf = conf;
        this.data = data;
        this.dateTimeData = null;
        this.userMappingData = userMappingData;
        this.itemMappingData = itemMappingData;
    } 

    public MatrixDataModel(Configuration conf, SparseMatrix data, SparseMatrix dateTimeData, BiMap<String, Integer> userMappingData, BiMap<String, Integer> itemMappingData){
        this.conf = conf;
        this.data = data;
        this.dateTimeData = dateTimeData;
        this.userMappingData = userMappingData;
        this.itemMappingData = itemMappingData;
    } 

    public DataSet getDatetimeDataSet() {
        return dataConvertor.getDatetimeMatrix();
    }

    public BiMap<String, Integer> getUserMappingData() {
        return userMappingData;
    }

    public BiMap<String, Integer> getItemMappingData() {
        return itemMappingData;
    }

    @Override
    protected void buildConvert() throws LibrecException {

        dataConvertor = new MatrixDataConvertor(data, dateTimeData);
        try {
            dataConvertor.processData();
        } catch (IOException e) {
            throw new LibrecException(e);
        }
    }

}