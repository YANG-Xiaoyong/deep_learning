package org.deeplearning.lossfun;

import java.util.List;

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
	 * m by n Matrix：预期结果下标
	 */
	private Integer[] expectedIndex;
	
	/**
	 * m by n Matrix：预期结果
	 */
	private List<Matrix> expectedResult;
	
	/**
	 * m by n Matrix：计算结果
	 */
	private Matrix computeResult;
	
	/**
	 * m by n Matrix：求导值
	 */
	private Matrix derivative;

	@Override
	public Double invoke() {
		// TODO Auto-generated method stub
		long count = this.computeResult.getRowCount();
		this.derivative = Matrix.Factory.ones(count, 1);
		
		double result = 0;
		for (int i = 0; i < count; i++) {
			//double diff = this.expectedResult.getAsDouble(expectedIndex[i],0) - this.computeResult.getAsDouble(i,0);
			double diff = this.expectedResult.get(expectedIndex[i]).getAsDouble(0,0) - this.computeResult.getAsDouble(i,0);
			result += diff * diff;
			this.derivative.setAsDouble(-2 * diff/count, i, 0);//设置导数
		}
		return result/count;
	}
	
	/*@Override
	public Double invoke() {
		// TODO Auto-generated method stub
		long count = this.computeResult.getRowCount();
		this.derivative = Matrix.Factory.ones(count, 1);
		
		double result = 0;
		for (int i = 0; i < count; i++) {
			double diff = this.expectedResult.getAsDouble(i,0) - this.computeResult.getAsDouble(i,0);
			result += diff * diff;
			this.derivative.setAsDouble(-2 * diff/count, i, 0);//设置导数
		}
		return result/count;
	}*/

	public List<Matrix> getExpectedResult() {
		return expectedResult;
	}

	public void setExpectedResult(List<Matrix> expectedResult) {
		this.expectedResult = expectedResult;
	}

	public Matrix getComputeResult() {
		return computeResult;
	}

	public void setComputeResult(Matrix computeResult) {
		this.computeResult = computeResult;
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

	public Integer[] getExpectedIndex() {
		return expectedIndex;
	}

	public void setExpectedIndex(Integer[] expectedIndex) {
		this.expectedIndex = expectedIndex;
	}
}
