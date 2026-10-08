import tensorflow as tf
import time
import os

from src.losses import AsymmetricLoss


def get_train_val_step(model, asl_loss, optimizer,
                       train_loss_metric, val_loss_metric, val_auc_metric):

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            preds = model(images, training=True)
            loss  = asl_loss(labels, preds)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss_metric.update_state(loss)

    @tf.function
    def val_step(images, labels):
        preds = model(images, training=False)
        loss  = asl_loss(labels, preds)
        val_loss_metric.update_state(loss)
        val_auc_metric.update_state(labels, preds)

    return train_step, val_step


def run_training(
    model,
    train_ds,
    val_ds,
    asl_loss,
    optimizer,
    train_loss_metric,
    val_loss_metric,
    val_auc_metric,
    epochs,
    save_dir,
    phase_name='phaseC',
    best_auc=0.0
):
    os.makedirs(save_dir, exist_ok=True)
    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}

    train_step, val_step = get_train_val_step(
        model, asl_loss, optimizer,
        train_loss_metric, val_loss_metric, val_auc_metric
    )

    print(f"Starting {phase_name} ({epochs} epochs)...")
    print("=" * 60)

    for epoch in range(epochs):
        start = time.time()
        train_loss_metric.reset_state()
        val_loss_metric.reset_state()
        val_auc_metric.reset_state()

        for step, (images, labels) in enumerate(train_ds):
            train_step(images, labels)
            if step % 200 == 0:
                print(f"  Epoch {epoch+1} | Step {step}/{len(train_ds)} | "
                      f"Loss: {train_loss_metric.result():.4f}")

        for images, labels in val_ds:
            val_step(images, labels)

        t_loss  = train_loss_metric.result().numpy()
        v_loss  = val_loss_metric.result().numpy()
        v_auc   = val_auc_metric.result().numpy()
        elapsed = time.time() - start

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['val_auc'].append(v_auc)

        print(f"\nEpoch {epoch+1}/{epochs} — "
              f"Train Loss: {t_loss:.4f} | "
              f"Val Loss: {v_loss:.4f} | "
              f"Val AUC: {v_auc:.4f} | "
              f"Time: {elapsed:.0f}s")

        # Save checkpoint every 5 epochs only
        if (epoch + 1) % 5 == 0:
            model.save(f'{save_dir}/{phase_name}_ep{epoch+1}.keras')
            print(f"  Checkpoint saved: {phase_name}_ep{epoch+1}.keras")

        if v_auc > best_auc:
            best_auc = v_auc
            model.save(f'{save_dir}/best_model_phaseC.keras')
            print(f"  ✓ Best model saved (AUC: {best_auc:.4f})")

        print("-" * 60)

    print(f"\n{phase_name} complete. Best AUC: {best_auc:.4f}")
    return history, best_auc