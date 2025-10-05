package com.example.graduated_project_lung

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.graphics.scale
import org.tensorflow.lite.Interpreter
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : AppCompatActivity() {

    private lateinit var interpreter: Interpreter
    private lateinit var imageView: ImageView
    private lateinit var resultText: TextView

    // استخدام Activity Result API مع Photo Picker
    private val pickMedia = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) {
            // تحميل الصورة باستخدام BitmapFactory
            val bitmap = loadBitmapFromUri(uri)
            if (bitmap != null) {
                imageView.setImageBitmap(bitmap)

                // تحويل الصورة لتناسب الموديل (224x224) باستخدام Bitmap.scale
                val scaledBitmap = bitmap.scale(224, 224, filter = false)
                val input = preprocessImage(scaledBitmap)
                val output = Array(1) { FloatArray(3) } // 3 فئات

                // تشغيل الموديل
                interpreter.run(input, output)

                // عرض النتيجة
                val result = getResult(output[0])
                resultText.text = result
            } else {
                Toast.makeText(this, "Failed to load image", Toast.LENGTH_SHORT).show()
            }
        } else {
            Toast.makeText(this, "No image selected", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // ربط العناصر من الـ UI
        imageView = findViewById(R.id.imageView)
        resultText = findViewById(R.id.resultText)
        val uploadButton: Button = findViewById(R.id.uploadButton)

        // تحميل الموديل من مجلد assets مع try-catch
        try {
            interpreter = Interpreter(loadModelFile())
        } catch (e: Exception) {
            e.printStackTrace()
            Toast.makeText(this, "Failed to load model: ${e.message}", Toast.LENGTH_LONG).show()
            return // توقف التنفيذ لو الموديل ما تحملش
        }

        // زرار رفع الصورة باستخدام Photo Picker
        uploadButton.setOnClickListener {
            // فتح Photo Picker
            pickMedia.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        }
    }

    // دالة لتحميل الصورة من URI
    private fun loadBitmapFromUri(uri: android.net.Uri): Bitmap? {
        return try {
            val inputStream: InputStream? = contentResolver.openInputStream(uri)
            BitmapFactory.decodeStream(inputStream).also { inputStream?.close() }
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    // دالة لتحميل ملف الموديل من assets
    private fun loadModelFile(): ByteBuffer {
        val assetManager = assets
        val fileDescriptor = assetManager.openFd("your_model.tflite") // تم تعديل الاسم هنا
        val inputStream = fileDescriptor.createInputStream()
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(java.nio.channels.FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    // تحويل الصورة لـ ByteBuffer
    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(4 * 224 * 224 * 3) // 4 bytes per float, 3 channels (RGB)
        byteBuffer.order(ByteOrder.nativeOrder())
        val pixels = IntArray(224 * 224)
        bitmap.getPixels(pixels, 0, 224, 0, 0, 224, 224)
        for (pixel in pixels) {
            byteBuffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f) // R
            byteBuffer.putFloat(((pixel shr 8) and 0xFF) / 250.0f)  // G
            byteBuffer.putFloat((pixel and 0xFF) / 255.0f)         // B
        }
        return byteBuffer
    }

    // تحليل نتيجة الموديل
    private fun getResult(output: FloatArray): String {
        val labels = arrayOf("lung_aca", "lung_n", "lung_scc")
        val maxIndex = output.indices.maxByOrNull { output[it] } ?: 0
        return "Prediction: ${labels[maxIndex]} (Confidence: ${output[maxIndex] * 100}%)"
    }
}