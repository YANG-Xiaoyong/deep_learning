package entity;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

public class LineLayer extends HiddenLayer {
	
	/**
	 * m by n Matrix
	 */
	private Matrix w;
	
	/**
	 * m by 1 Matrix：Vetor
	 */
	private Matrix b;
	
	/**
	 * m by 1 Matrix：derivative
	 */
	private Matrix derivative;
	
	@Override
	public Matrix calculate() {
		// TODO Auto-generated method stub
		return this.w.mtimes(this.x).plus(this.b);
	}
	
	/**
	 * 随机生成矩阵w和b
	 * @param rows
	 * @param cols
	 * @return
	 */
	public LineLayer setWbRandom(long rows, long cols) {
		if(this.w == null) {
			this.w = Matrix.Factory.ones(rows, cols);
			this.w.randn(Ret.ORIG);
			this.b = Matrix.Factory.ones(rows, 1);
			this.b.randn(Ret.ORIG);
		}
		return this;
	}

	public Matrix getW() {
		return w;
	}


	public void setW(Matrix w) {
		this.w = w;
	}


	public Matrix getB() {
		return b;
	}


	public void setB(Matrix b) {
		this.b = b;
	}

	public Matrix getDerivative() {
		return derivative;
	}

	public void setDerivative(Matrix derivative) {
		this.derivative = derivative;
	}
}
