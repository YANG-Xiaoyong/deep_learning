package org.deeplearning.activeFun;

import org.ujmp.core.Matrix;

/**
 * 恒等激活函数，用于隐层神经元输出。公式如下
 * f(x) = x
 * @author yangxy
 *
 */
public class IdenticalActiveFun extends AbstractActiveFun {
	
	@Override
	public void calculateResult(Matrix resultSet) {
		// TODO Auto-generated method stub
	}

	@Override
	public void calculateDerivative(Matrix reluDerivative, Matrix resultSet) {
		// TODO Auto-generated method stub
		for(int rows = 0; rows < resultSet.getRowCount(); rows++) {
			for(int cols = 0; cols < resultSet.getColumnCount(); cols++) {
				if(resultSet.getAsDouble(rows, cols) > 0) {
					reluDerivative.setAsDouble(1, rows, cols);
				} else {
					reluDerivative.setAsDouble(-1, rows, cols);
				}
			}
		}
		
	}
	
	

}
