package org.deeplearning.activeFun;

import org.ujmp.core.Matrix;

public abstract class AbstractActiveFun {
	
	public abstract void calculateResult(Matrix resultSet);
	
	public abstract void calculateDerivative(Matrix reluDerivative, Matrix resultSet);
	
}
