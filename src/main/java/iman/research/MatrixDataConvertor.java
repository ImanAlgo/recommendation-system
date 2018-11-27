package iman.research;

import java.io.IOException;

import net.librec.data.convertor.AbstractDataConvertor;
import net.librec.math.structure.SparseMatrix;

public class MatrixDataConvertor extends AbstractDataConvertor {

    public MatrixDataConvertor(SparseMatrix data){
        this.preferenceMatrix = data;
    }

    public MatrixDataConvertor(SparseMatrix data, SparseMatrix dateTimeData){
        this.preferenceMatrix = data;
        this.datetimeMatrix = dateTimeData;
    }

    public void processData() throws IOException {

    }

    public void progress() {
		
	}

}