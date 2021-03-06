package org.deeplearning.activeFun;

import org.ujmp.core.Matrix;

/**
 * Relu激活函数（The Rectified Linear Unit），用于隐层神经元输出。公式如下
 * f(x) = max(0,x)
 * @author yangxy
 *
 */
public class ReLUActiveFun extends AbstractActiveFun {
	
	@Override
	public void calculateResult(Matrix resultSet) {
		// TODO Auto-generated method stub
		long count = resultSet.getRowCount();
		for (int i = 0; i < count; i++) {
			if(resultSet.getAsDouble(i,0) < 0) {
				resultSet.setAsDouble(0,i,0);
			}
		}
	}

	@Override
	public void calculateDerivative(Matrix reluDerivative, Matrix resultSet) {
		// TODO Auto-generated method stub
		for(int rows = 0; rows < resultSet.getRowCount(); rows++) {
			for(int cols = 0; cols < resultSet.getColumnCount(); cols++) {
				if(resultSet.getAsDouble(rows, cols) > 0) {
					reluDerivative.setAsDouble(1, rows, cols);
				} else {
					reluDerivative.setAsDouble(0, rows, cols);
				}
			}
		}
		
	}
	
	

}
