package org.deeplearning.lossfun;

import org.ujmp.core.Matrix;

/**
 * 均方误差
 * @author yangxy
 *
 */
public class MseLostFun extends AbstractLostFun {
	
	/**
	 * 阈值
	 */
	private Double threshold = 0.0001;
	
	/**
	 * m by n Matrix：测试集
	 */
	private Matrix testSet;
	
	/**
	 * m by n Matrix：结果集
	 */
	private Matrix resultSet;
	
	/**
	 * m by n Matrix：求导值
	 */
	private Matrix derivative;

	@Override
	public Double invoke() {
		// TODO Auto-generated method stub
		long count = this.resultSet.getRowCount();
		this.derivative = Matrix.Factory.ones(count, 1);
		
		double result = 0;
		for (int i = 0; i < count; i++) {
			double diff = this.testSet.getAsDouble(i,0) - this.resultSet.getAsDouble(i,0);
			result += diff * diff;
			this.derivative.setAsDouble(-2 * diff/count, i, 0);//设置导数
		}
		return result/count;
	}

	public Matrix getTestSet() {
		return testSet;
	}

	public void setTestSet(Matrix testSet) {
		this.testSet = testSet;
	}

	public Matrix getResultSet() {
		return resultSet;
	}

	public void setResultSet(Matrix resultSet) {
		this.resultSet = resultSet;
	}

	public Double getThreshold() {
		return threshold;
	}

	public void setThreshold(Double threshold) {
		this.threshold = threshold;
	}

	public Matrix getDerivative() {
		return derivative;
	}

	public void setDerivative(Matrix derivative) {
		this.derivative = derivative;
	}
}
