import SwiftUI
import UIKit
import TensorFlowLite
import Accelerate

struct ContentView: View {
    @State private var selectedImage: UIImage? = nil
    @State private var showImagePicker = false
    var tempImage:Data? = nil
    
    var body: some View {
        VStack {
            Text("Select an Image")
                .font(.headline)
                .padding()
            
            if let image = selectedImage {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(height: 400)
                    .cornerRadius(10)
            } else {
                Rectangle()
                    .fill(Color.gray.opacity(0.2))
                    .frame(height: 400)
                    .cornerRadius(10)
                    .overlay(Text("No Image").foregroundColor(.gray))
            }
            
            Button(action: {
                showImagePicker.toggle()
            }) {
                Text("Open Image Picker")
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
            }
            .padding()
            
            Button(action: {
                print("Remove button click")
                runModel(on: selectedImage)
            }) {
                Text("Remove Background")
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
            }
            .padding()
        }
        .sheet(isPresented: $showImagePicker) {
            ImagePicker(image: $selectedImage)
        }
        .padding()
    }
    
    // 加载和运行 TensorFlow Lite 模型
    func runModel(on image: UIImage?) -> String {
        // 加载模型
        guard let modelPath = Bundle.main.path(forResource: "tf_u2netp_model", ofType: "tflite") else {
            print("Model not found")
            return "Model not found"
        }
        
        do {
            let interpreter = try Interpreter(modelPath: modelPath)
            try interpreter.allocateTensors()
            
            let inputTensor = try interpreter.input(at: 0)
            
            let resizeImage = resizeUIImage(image: selectedImage!, width: 320, height: 320)
            let inputData = transformDimensions(image: resizeImage!, n: 1, w: 320, h: 320, c: 3)
            try interpreter.copy(inputData!, toInputAt: 0)
            
            try interpreter.invoke()
            
            let outputTensor = try interpreter.output(at: 0)
            let outputData = outputTensor.data
            print("Output shape: \(outputTensor.shape)")
            
            selectedImage = modifyAlphaAtPoint(output: outputData,oriImage: selectedImage!)
            
            return "OK"
        } catch {
            print("Error running model: \(error)")
            return "Error"
        }
    }
    
    func resizeUIImage(
        image:UIImage,
        width:Int,
        height:Int
    ) -> UIImage? {
        let targetSize = CGSize(
            width: width,
            height: height
        )
        UIGraphicsBeginImageContextWithOptions(
            targetSize,
            false,
            1.0
        )
        image.draw(
            in: CGRect(
                origin: .zero,
                size: targetSize
            )
        )
        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return resizedImage
    }
    
    
    func transformDimensions(image: UIImage, n: Int, w: Int, h: Int, c: Int) -> Data? {
        guard let cgImage = image.cgImage else {
            return nil
        }
        
        let width = cgImage.width
        let height = cgImage.height
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let bitsPerComponent = 8
        let imageData = UnsafeMutablePointer<UInt8>.allocate(capacity: width * height * bytesPerPixel)
        defer { imageData.deallocate() }
        
        guard let context = CGContext(data: imageData,
                                      width: width,
                                      height: height,
                                      bitsPerComponent: bitsPerComponent,
                                      bytesPerRow: bytesPerRow,
                                      space: colorSpace,
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            return nil
        }
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        var floatArray = [Float](repeating: 0, count: width * height * c)
        for i in 0..<width * height {
            for j in 0..<c {
                floatArray[i * c + j] = Float(imageData[i * bytesPerPixel + j]) / 255.0
            }
        }
        
        let newSize = n * c * w * h
        var transformed = [Float](repeating: 0, count: newSize)
        
        for i in 0..<n {
            for j in 0..<w {
                for k in 0..<h {
                    for l in 0..<c {
                        let oldIndex = i * (w * h * c) + j * (h * c) + k * c + l
                        let newIndex = i * (c * w * h) + l * (w * h) + j * h + k
                        transformed[newIndex] = floatArray[oldIndex]
                    }
                }
            }
        }
        
        let data = transformed.withUnsafeBufferPointer { Data(buffer: $0) }
        return data
    }
    
    func resizeGrayscaleImage(input: [Float], originalWidth: Int, originalHeight: Int, newWidth: Int, newHeight: Int) -> [Float] {
        var output = [Float](repeating: 0.0, count: newWidth * newHeight)
        let xScale = Double(originalWidth) / Double(newWidth)
        let yScale = Double(originalHeight) / Double(newHeight)
        
        for y in 0..<newHeight {
            for x in 0..<newWidth {
                let srcX = Int(Double(x) * xScale)
                let srcY = Int(Double(y) * yScale)
                let srcIndex = srcY * originalWidth + srcX
                output[y * newWidth + x] = input[srcIndex]
            }
        }
        
        return output
    }
    
    func modifyAlphaAtPoint(output:Data, oriImage:UIImage) -> UIImage? {
        guard let cgImage = oriImage.cgImage else { return nil }
        
        let floatArray: [Float] = output.withUnsafeBytes { Array(UnsafeBufferPointer<Float>(start: $0.bindMemory(to: Float.self).baseAddress!, count: output.count / MemoryLayout<Float>.size)) }
        
        let width = cgImage.width
        let height = cgImage.height
        print("width: \(width), height: \(height)")
        let resizeMask = resizeGrayscaleImage(input: floatArray, originalWidth: 320, originalHeight: 320, newWidth: width, newHeight: height)
        
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let bitmapData = UnsafeMutableRawPointer.allocate(byteCount: height * bytesPerRow, alignment: MemoryLayout<UInt8>.alignment)
        
        guard let context = CGContext(
            data: bitmapData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue) else {
            fatalError("无法创建图形上下文")
        }
        
        context.draw(oriImage.cgImage!, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        guard let pixelData = context.data else { return nil }
        
        let data = pixelData.bindMemory(to: UInt8.self, capacity: width * height * 4)
        
        for y in 0..<height {
            for x in 0..<width {
                let index = (y * width + x)
                let alphaValue = CGFloat(resizeMask[index]) // 获取输出数据作为 alpha 值
                data[index * 4 + 3] = UInt8(alphaValue * 255)
            }
        }
        
        let ci = context.makeImage()
        return UIImage(cgImage: ci!)
    }
}

// iOS 图片选择器
struct ImagePicker: UIViewControllerRepresentable {
    @Binding var image: UIImage?
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.delegate = context.coordinator
        picker.sourceType = .photoLibrary
        return picker
    }
    
    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}
    
    class Coordinator: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
        var parent: ImagePicker
        
        init(_ parent: ImagePicker) {
            self.parent = parent
        }
        
        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            if let uiImage = info[.originalImage] as? UIImage {
                parent.image = uiImage
            }
            picker.dismiss(animated: true)
        }
        
        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            picker.dismiss(animated: true)
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}



