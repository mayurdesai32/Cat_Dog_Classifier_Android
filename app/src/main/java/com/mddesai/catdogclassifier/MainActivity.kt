package com.mddesai.catdogclassifier
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import com.mddesai.catdogclassifier.ml.TfLiteQuantModel
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class MainActivity : AppCompatActivity() {
    private lateinit var bitmap: Bitmap
    private lateinit var imageView: ImageView
    private lateinit var outputText: TextView
    private lateinit var model: TfLiteQuantModel

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)


        imageView = findViewById(R.id.image)
        val galleryButton = findViewById<Button>(R.id.gallery)
        val predictButton = findViewById<Button>(R.id.predict)
        outputText = findViewById(R.id.outputText)

        // Initialize the TensorFlow Lite model
        model = TfLiteQuantModel.newInstance(this)

        val imageProcessor = ImageProcessor.Builder()
            .add(NormalizeOp(0.0f, 255.0f))
            .add(ResizeOp(256, 256, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
            .build()

        galleryButton.setOnClickListener {
            val intent = Intent(Intent.ACTION_GET_CONTENT)
            intent.type = "image/*"
            startActivityForResult(intent, REQUEST_CODE_PICK_IMAGE)
        }

        predictButton.setOnClickListener {
            val tensorImage = TensorImage(DataType.FLOAT32)
            tensorImage.load(bitmap)


            val processedImage = imageProcessor.process(tensorImage)

            // Create input tensor buffer
            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 256, 256, 3), DataType.FLOAT32)
            inputFeature0.loadBuffer(processedImage.buffer)

            // Perform model inference
            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer.floatArray

            // Determine the predicted class
            val maxIdx = outputFeature0.indices.maxByOrNull { outputFeature0[it] } ?: -1
            val outputText1 = if (maxIdx != -1) {
                "Predicted class: ${if (maxIdx == 0) "Cat" else "Dog"}"
            } else {
                "Prediction failed"
            }

            // Display the result
            outputText.text = outputText1
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQUEST_CODE_PICK_IMAGE && resultCode == RESULT_OK) {
            val uri = data?.data
            uri?.let {
                try {
                    bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)


                    val options = BitmapFactory.Options()
                    options.inSampleSize = 2 // Adjust the sample size as needed to reduce image size
                   val scaleBitmap = BitmapFactory.decodeStream(contentResolver.openInputStream(uri), null, options)




                    imageView.setImageBitmap(scaleBitmap)
                } catch (e: Exception) {
                    e.printStackTrace()
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        // Release model resources
        model.close()
    }

    companion object {
        private const val REQUEST_CODE_PICK_IMAGE = 100
    }
}
