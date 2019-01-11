package org.deeplearning.layer;

import org.ujmp.core.Matrix;

public abstract class AbstractHiddenLayer {

	/**
	 * m by n Matrix：input
	 */
	public Matrix x;
	
	public Matrix getX() {
		return x;
	}

	public void setX(Matrix x) {
		this.x = x;
	}
	
	abstract Matrix calculate();
	
}
