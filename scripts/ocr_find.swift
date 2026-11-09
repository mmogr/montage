// ocr_find VIDEO PATTERN [step_seconds]
// Samples frames from VIDEO and prints the seconds where on-screen text matches PATTERN (case-insensitive regex).
import AVFoundation
import Vision
import Foundation

let args = CommandLine.arguments
guard args.count >= 3 else { print("usage: ocr_find VIDEO PATTERN [step]"); exit(2) }
let url = URL(fileURLWithPath: args[1])
let pattern = try! NSRegularExpression(pattern: args[2], options: [.caseInsensitive])
let step = args.count > 3 ? Double(args[3])! : 1.0

let asset = AVURLAsset(url: url)
let gen = AVAssetImageGenerator(asset: asset)
gen.appliesPreferredTrackTransform = true
gen.requestedTimeToleranceBefore = CMTime(seconds: 0.25, preferredTimescale: 600)
gen.requestedTimeToleranceAfter = CMTime(seconds: 0.25, preferredTimescale: 600)

let sem = DispatchSemaphore(value: 0)
var duration = 0.0
Task { duration = (try? await asset.load(.duration).seconds) ?? 0; sem.signal() }
sem.wait()

var t = 0.0
while t < duration {
    if let img = try? gen.copyCGImage(at: CMTime(seconds: t, preferredTimescale: 600), actualTime: nil) {
        let req = VNRecognizeTextRequest()
        req.recognitionLevel = .accurate
        try? VNImageRequestHandler(cgImage: img).perform([req])
        let texts = (req.results ?? []).compactMap { $0.topCandidates(1).first?.string }
        for s in texts where pattern.firstMatch(in: s, range: NSRange(s.startIndex..., in: s)) != nil {
            print(String(format: "%.1f\t%@", t, s))
        }
    }
    t += step
}
