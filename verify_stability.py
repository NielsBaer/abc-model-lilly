
import jax.numpy as jnp
import numpy as np
import abcconfigs.class_model as cm
import abcmodel
import sys

# disable jit to debug values
import jax
jax.config.update("jax_disable_jit", True)

def main():
    print("Verifying wstar stability...")
    
    # Instantiate mixed layer model
    mixed_layer_model = abcmodel.atmosphere.mixed_layer.BulkMixedLayerModel(
        **cm.bulk_mixed_layer.model_kwargs,
    )
    
    # Test compute_wstar with tiny positive flux
    h_abl = 1000.0
    thetav = 300.0
    g = 9.81
    
    # Case 1: Tiny positive wthetav (e.g. 1e-12) -> should be clamped to 1e-6
    # Theoretical wstar = (g * h * wthetav / thetav)^(1/3)
    # (9.81 * 1000 * 1e-12 / 300)^(1/3) approx (3e-11)^(1/3) approx 3e-4
    # Wait, 3e-4 > 1e-6.
    # Let's try even smaller. 1e-20.
    # (3e-19)^(1/3) approx 6e-7 < 1e-6.
    
    wthetav_tiny = 1e-20
    wstar = mixed_layer_model.compute_wstar(h_abl, wthetav_tiny, thetav, g)
    print(f"Input wthetav: {wthetav_tiny}")
    print(f"Output wstar: {wstar}")
    
    if wstar < 1e-6:
        print("FAILURE: wstar was not clamped to 1e-6!")
        sys.exit(1)
        
    # Case 2: Negative wthetav -> should be 1e-6 (existing logic, but good to check)
    wthetav_neg = -1.0
    wstar_neg = mixed_layer_model.compute_wstar(h_abl, wthetav_neg, thetav, g)
    print(f"Input wthetav (neg): {wthetav_neg}")
    print(f"Output wstar: {wstar_neg}")
    
    if wstar_neg < 1e-6:
         print("FAILURE: wstar (negative flux) was not clamped to 1e-6!")
         sys.exit(1)

    print("\nVerifying Cloud Model stability with wstar=1e-6...")
    cloud_model = abcmodel.atmosphere.clouds.StandardCumulusModel()
    
    # Create valid inputs for compute_q2_h (which crashed previously)
    cc_qf = 0.0
    wqe = 0.0
    dq = -0.001
    dz_h = 200.0
    
    # Testing compute_q2_h
    # compute_q2_h divides by wstar
    wstar_min = 1e-6
    q2_h = cloud_model.compute_q2_h(cc_qf, 1.0, wqe, dq, h_abl, dz_h, wstar_min)
    print(f"q2_h with wstar={wstar_min}: {q2_h}")
    
    if not jnp.isfinite(q2_h):
        print("FAILURE: q2_h is not finite!")
        sys.exit(1)

    print("SUCCESS: wstar is stable and cloud model accepts minimum wstar.")

if __name__ == "__main__":
    main()
