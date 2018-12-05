package iman.research;

import net.librec.common.LibrecException;
import net.librec.data.splitter.AbstractDataSplitter;
import net.librec.math.structure.SparseMatrix;

public class PreSplitedDataSplitter extends AbstractDataSplitter {

    public PreSplitedDataSplitter(SparseMatrix trainData, SparseMatrix testData){
        this.trainMatrix = trainData;
        this.testMatrix = testData;
    }

    public void splitData() throws LibrecException {
        // NO OPERATION
	}

}