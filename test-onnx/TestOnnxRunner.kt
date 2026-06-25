import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.io.File
import kotlin.math.exp

fun main() {
    val ortEnvironment = OrtEnvironment.getEnvironment()
    val modelBytes = File("../android-app/RoadAnomalyDetector/app/src/main/assets/model_1dcnn.onnx").readBytes()
    val options = OrtSession.SessionOptions()
    val ortSession = ortEnvironment.createSession(modelBytes, options)

    val flatData = FloatArray(600)
    flatData[0] = 10.0f
    flatData[200] = 2.0f
    flatData[400] = 1.45f

    val shape = longArrayOf(1, 3, 200)

    val byteBuffer = java.nio.ByteBuffer.allocateDirect(flatData.size * 4)
    byteBuffer.order(java.nio.ByteOrder.nativeOrder())
    val floatBuffer = byteBuffer.asFloatBuffer()
    floatBuffer.put(flatData)
    floatBuffer.rewind()

    val tensor = OnnxTensor.createTensor(ortEnvironment, floatBuffer, shape)
    
    val inputName = ortSession.inputNames.iterator().next()
    val inputs = mapOf(inputName to tensor)
    
    val result = ortSession.run(inputs)
    
    val outputTensor = result.iterator().next().value as OnnxTensor
    val outFloatBuffer = outputTensor.floatBuffer
    val logits = FloatArray(3)
    outFloatBuffer.get(logits)
    
    println("Logits: ${logits.joinToString()}")
    
    var maxLogit = Float.NEGATIVE_INFINITY
    for (l in logits) {
        if (l > maxLogit) maxLogit = l
    }
    
    var sumExp = 0f
    val expLogits = FloatArray(logits.size)
    for (i in logits.indices) {
        val e = exp((logits[i] - maxLogit).toDouble()).toFloat()
        expLogits[i] = e
        sumExp += e
    }
    
    for (i in logits.indices) {
        expLogits[i] = expLogits[i] / sumExp
    }
    
    println("Probs: ${expLogits.joinToString()}")
}
