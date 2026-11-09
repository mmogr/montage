// ocr_storyboard SHEET_DIR COLS ROWS INTERVAL_SECONDS
// Splits YouTube storyboard sheets (sorted *.jpg) into frames and OCRs every frame.
// Prints one line per recognized text line: "<seconds>\t<confidence>\t<text>".
import Foundation
import Vision
import ImageIO

let a = CommandLine.arguments
guard a.count == 5, let cols = Int(a[2]), let rows = Int(a[3]), let interval = Double(a[4]) else {
    FileHandle.standardError.write("usage: ocr_storyboard SHEET_DIR COLS ROWS INTERVAL\n".data(using: .utf8)!); exit(2)
}
let dir = URL(fileURLWithPath: a[1])
let sheets = (try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil))
    .filter { $0.pathExtension == "jpg" }.sorted { $0.lastPathComponent < $1.lastPathComponent }

var out = [Int: [String]]()
let lock = NSLock()
DispatchQueue.concurrentPerform(iterations: sheets.count) { si in
    guard let src = CGImageSourceCreateWithURL(sheets[si] as CFURL, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return }
    let tw = img.width / cols, th = img.height / rows
    var lines = [String]()
    for r in 0..<rows { for c in 0..<cols {
        let k = si * cols * rows + r * cols + c
        guard let tile = img.cropping(to: CGRect(x: c * tw, y: r * th, width: tw, height: th)) else { continue }
        let req = VNRecognizeTextRequest()
        req.recognitionLevel = .accurate
        req.usesLanguageCorrection = false
        try? VNImageRequestHandler(cgImage: tile).perform([req])
        for o in req.results ?? [] {
            if let cand = o.topCandidates(1).first {
                lines.append(String(format: "%.1f\t%.2f\t%@", Double(k) * interval, cand.confidence,
                                    cand.string.replacingOccurrences(of: "\t", with: " ")))
            }
        }
    }}
    lock.lock(); out[si] = lines; lock.unlock()
}
for si in out.keys.sorted() { for l in out[si]! { print(l) } }
