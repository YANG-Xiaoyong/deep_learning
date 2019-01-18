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
	public Matrix y;
	
	
	/**
	 * m by 1 Matrix：derivative
	 */
	public Matrix derivative;
	
	/**
	 * m by n Matrix
	 */
	public Matrix w;
	
	/**
	 * m by 1 Matrix：Vetor
	 */
	public Matrix b;
	
	/**
	 * 宽度
	 */
	public Long width;
	
	public AbstractHiddenLayer() {
	}
	
	public AbstractHiddenLayer(Long width) {
		this.width = width;
	}
	
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

	public long getWidth() {
		return width;
	}

	public void setWidth(long width) {
		this.width = width;
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

	public void setWidth(Long width) {
		this.width = width;
	}

	abstract Matrix calculate();
	
}
