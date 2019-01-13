package org.deeplearning.actveFun;

import org.ujmp.core.Matrix;

/**
 * Relu激活函数（The Rectified Linear Unit），用于隐层神经元输出。公式如下
 * f(x) = max(0,x)
 * @author yangxy
 *
 */
public class ReLUActiveFun extends AbstractActiveFun {
	
	@Override
	public void invoke(Matrix resultSet) {
		// TODO Auto-generated method stub
		long count = resultSet.getRowCount();
		for (int i = 0; i < count; i++) {
			if(resultSet.getAsDouble(i,0) < 0) {
				resultSet.setAsDouble(0,i,0);
			}
		}
	}

}
