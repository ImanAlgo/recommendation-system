package iman.research;

import java.io.IOException;

import com.google.common.collect.BiMap;

import net.librec.common.LibrecException;
import net.librec.conf.Configuration;
import net.librec.data.DataModel;
import net.librec.data.model.AbstractDataModel;
import net.librec.math.structure.DataSet;
import net.librec.math.structure.SparseMatrix;

public class MatrixDataModel extends AbstractDataModel implements DataModel {

    private SparseMatrix data;
    private SparseMatrix dateTimeData;
    private BiMap<String, Integer> userMappingData;
    private BiMap<String, Integer> itemMappingData;

    public MatrixDataModel(SparseMatrix trainData, SparseMatrix testData, BiMap<String, Integer> userMappingData, BiMap<String, Integer> itemMappingData){
        this.conf = new Configuration();
        this.data = null;
        this.dateTimeData = null;
        this.userMappingData = userMappingData;
        this.itemMappingData = itemMappingData;

        this.dataSplitter = new PreSplitedDataSplitter(trainData, testData);
    } 

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
        if(data != null){
            dataConvertor = new MatrixDataConvertor(data, dateTimeData);
            try {
                dataConvertor.processData();
            } catch (IOException e) {
                throw new LibrecException(e);
            }
        }
    }

}