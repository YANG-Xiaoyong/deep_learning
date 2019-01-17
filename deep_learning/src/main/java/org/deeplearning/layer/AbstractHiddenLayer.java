package org.deeplearning.layer;

import org.ujmp.core.Matrix;

public abstract class AbstractHiddenLayer {

	/**
	 * m by n Matrix：input
	 */
	public Matrix x;
	
	/**
	 * m by n Matrix：output
	 */
	private Matrix y;
	
	
	/**
	 * m by 1 Matrix：derivative
	 */
	private Matrix derivative;
	
	
	public Matrix getX() {
		return x;
	}

	public void setX(Matrix x) {
		this.x = x;
	}
	
	public Matrix getY() {
		return y;
	}

	public void setY(Matrix y) {
		this.y = y;
	}

	public Matrix getDerivative() {
		return derivative;
	}

	public void setDerivative(Matrix derivative) {
		this.derivative = derivative;
	}

	abstract Matrix calculate();
	
}
