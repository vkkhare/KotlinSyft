package org.openmined.syft.proto

import android.util.Log
import org.openmined.syft.R
import org.openmined.syftproto.execution.v1.StateOuterClass
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File
import java.net.URI

private const val TAG = "SyftModel"

/**
 * SyftModel is the data model class for storing the weights of the neural network used for training or inference.
 * @property modelName A string to hold the name of the model specified while hosting the plan on PyGrid.
 * @property version  A string specifying the version of the model.
 * @property pyGridModelId  A unique id assigned by PyGrid to very model hosted over it used for downloading model weights
 * @property modelState Responsible for holding the latest model weights of the neural network.
 * @property startState Responsible for holding the model weights given by PyGrid.
 * @see createDiff for usage of startState
 */
@ExperimentalUnsignedTypes
data class SyftModel(
    val modelName: String,
    val version: String? = null,
    var pyGridModelId: String? = null,
    var modelState: State? = null,
    var startState: State? = null
) {

    /**
     * This method is used to save/update SyftModel class.
     * This function must be called after every gradient step to update the model state for
     * further plan executions.
     *
     * @param newModelParams a list of pyTorch Tensor that would be converted to syftTensor
     * @sample updateModel(updatedParams.map { it.toTensor() })
     */
    fun updateModel(newModelParams: List<Tensor>) {
        modelState?.let { state ->
            newModelParams.forEachIndexed { index, pytorchTensor ->
                state.syftTensors[index] - SyftTensor.fromTorchTensor(pytorchTensor)
                state.syftTensors[index] = SyftTensor.fromTorchTensor(pytorchTensor)
            }
        }
    }

    /**
     * This method is used to load SyftModel from protobuff file
     *
     * @param modelFile the filepath containing model state.
     * @sample loadModelState("modelFile")
     */
    fun loadModelState(modelFile: String) {
        startState = State.deserialize(
            StateOuterClass.State.parseFrom(File(modelFile).readBytes())
        )
        modelState = startState
        Log.d(TAG, "Model loaded from $modelFile")
    }
}

