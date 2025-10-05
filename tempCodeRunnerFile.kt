package com.example.graduated_project_lung

import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Typeface
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.text.Spannable
import android.text.SpannableString
import android.text.style.ForegroundColorSpan
import android.text.style.StyleSpan
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.core.graphics.scale
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity() {

    private lateinit var interpreter: Interpreter
    private lateinit var imageView: ImageView
    private lateinit var resultText: TextView
    private lateinit var resultDescription: TextView
    private lateinit var progressBar: ProgressBar
    private lateinit var photoURI: Uri

    private val pickMedia = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) {
            processImage(uri)
        } else {
            Toast.makeText(this, "No image selected", Toast.LENGTH_SHORT).show()
        }
    }

    private val takePicture = registerForActivityResult(ActivityResultContracts.TakePicture()) { success ->
        if (success) {
            processImage(photoURI)
        } else {
            Toast.makeText(this, "Failed to capture image", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.imageView)
        resultText = findViewById(R.id.resultText)
        resultDescription = findViewById(R.id.resultDescription)
        progressBar = findViewById(R.id.progressBar)
        val uploadButton: Button = findViewById(R.id.uploadButton)
        val takePhotoButton: Button = findViewById(R.id.takePhotoButton)

        try {
            interpreter = Interpreter(loadModelFile())
        } catch (e: Exception) {
            e.printStackTrace()
            Toast.makeText(this, "Failed to load model: ${e.message}", Toast.LENGTH_LONG).show()
            return
        }

        uploadButton.setOnClickListener {
            pickMedia.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        }

        takePhotoButton.setOnClickListener {
            if (checkCameraPermission()) {
                dispatchTakePictureIntent()
            } else {
                requestCameraPermission()
            }
        }
    }

    private fun processImage(uri: Uri) {
        progressBar.visibility = View.VISIBLE

        thread {
            val bitmap = loadBitmapFromUri(uri)
            if (bitmap != null) {
                runOnUiThread {
                    imageView.setImageBitmap(bitmap)
                }

                val scaledBitmap = bitmap.scale(224, 224, filter = false)
                val input = preprocessImage(scaledBitmap)
                val output = Array(1) { FloatArray(3) }

                interpreter.run(input, output)

                runOnUiThread {
                    val result = getResult(output[0])
                    val fullResultText = result
                    val resultLabel = result.substringAfter("Prediction: ").substringBefore(" (")

                    val resultColor = when (resultLabel) {
                        "Lung_benign_tissue" -> R.color.safeColor
                        in listOf("Lung_adenocarcinoma", "Lung squamous_cell_carcinoma") -> R.color.dangerColor
                        else -> R.color.colorPrimary
                    }

                    val spannable = SpannableString(fullResultText)
                    val startIndex = fullResultText.indexOf(resultLabel)
                    val endIndex = startIndex + resultLabel.length

                    spannable.setSpan(
                        ForegroundColorSpan(ContextCompat.getColor(this, resultColor)),
                        startIndex,
                        endIndex,
                        Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
                    )
                    spannable.setSpan(
                        StyleSpan(Typeface.BOLD),
                        startIndex,
                        endIndex,
                        Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
                    )

                    resultText.text = spannable
                    resultDescription.text = getResultDescription(output[0])
                    progressBar.visibility = View.GONE
                }
            } else {
                runOnUiThread {
                    Toast.makeText(this, "Failed to load image", Toast.LENGTH_SHORT).show()
                    progressBar.visibility = View.GONE
                }
            }
        }
    }

    private fun loadBitmapFromUri(uri: Uri): Bitmap? {
        return try {
            val inputStream: InputStream? = contentResolver.openInputStream(uri)
            BitmapFactory.decodeStream(inputStream).also { inputStream?.close() }
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    private fun loadModelFile(): ByteBuffer {
        val assetManager = assets
        val fileDescriptor = assetManager.openFd("your_model.tflite")
        val inputStream = fileDescriptor.createInputStream()
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(java.nio.channels.FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(4 * 224 * 224 * 3)
        byteBuffer.order(ByteOrder.nativeOrder())
        val pixels = IntArray(224 * 224)
        bitmap.getPixels(pixels, 0, 224, 0, 0, 224, 224)
        for (pixel in pixels) {
            byteBuffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f)
            byteBuffer.putFloat(((pixel shr 8) and 0xFF) / 250.0f)
            byteBuffer.putFloat((pixel and 0xFF) / 255.0f)
        }
        return byteBuffer
    }

    private fun getResult(output: FloatArray): String {
        val labels = arrayOf("Lung_adenocarcinoma", "Lung_benign_tissue", "Lung squamous_cell_carcinoma")
        val maxIndex = output.indices.maxByOrNull { output[it] } ?: 0
        val confidence = output[maxIndex] * 100

        return when {
            confidence < 95 -> "Invalid Image: This image does not appear to be a valid lung scan."
            confidence in 95.0..97.0 -> {
                "Prediction: ${labels[maxIndex]} (Confidence: ${String.format(Locale.US, "%.2f", confidence)}%) - There is some uncertainty in this result. Please consult a specialist."
            }
            else -> "Prediction: ${labels[maxIndex]} (Confidence: ${String.format(Locale.US, "%.2f", confidence)}%)"
        }
    }

    private fun getResultDescription(output: FloatArray): String {
        val labels = arrayOf("Lung_adenocarcinoma", "Lung_benign_tissue", "Lung squamous_cell_carcinoma")
        val maxIndex = output.indices.maxByOrNull { output[it] } ?: 0
        val confidence = output[maxIndex] * 100

        return when {
            confidence < 95 -> "Please upload a proper lung scan image for accurate results."
            confidence in 95.0..97.0 -> {
                val baseDescription = when (labels[maxIndex]) {
                    "Lung_adenocarcinoma" -> "This may indicate Lung adenocarcinoma, a type of lung cancer."
                    "Lung_benign_tissue" -> "This indicates a benign lung tissue."
                    "Lung squamous_cell_carcinoma" -> "This may indicate Lung squamous cell carcinoma, a type of lung cancer."
                    else -> "Unknown result."
                }
                "$baseDescription Please consult a specialist for confirmation."
            }
            else -> when (labels[maxIndex]) {
                "Lung_adenocarcinoma" -> "This may indicate Lung adenocarcinoma, a type of lung cancer. Please consult a doctor."
                "Lung_benign_tissue" -> "This indicates a benign lung tissue. However, regular checkups are recommended."
                "Lung squamous_cell_carcinoma" -> "This may indicate Lung squamous cell carcinoma, a type of lung cancer. Please consult a doctor."
                else -> "Unknown result."
            }
        }
    }

    private fun checkCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            this,
            android.Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun requestCameraPermission() {
        ActivityCompat.requestPermissions(
            this,
            arrayOf(android.Manifest.permission.CAMERA),
            CAMERA_PERMISSION_CODE
        )
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_PERMISSION_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                dispatchTakePictureIntent()
            } else {
                Toast.makeText(this, "Camera permission denied", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun createImageFile(): File {
        val timeStamp: String = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
        val storageDir: File? = getExternalFilesDir(Environment.DIRECTORY_PICTURES)
        return File.createTempFile("JPEG_${timeStamp}_", ".jpg", storageDir)
    }

    private fun dispatchTakePictureIntent() {
        val photoFile: File = createImageFile()
        photoURI = FileProvider.getUriForFile(this, "${packageName}.fileprovider", photoFile)
        takePicture.launch(photoURI)
    }

    companion object {
        private const val CAMERA_PERMISSION_CODE = 100
    }
}