//
//  Project.swift
//  StopDistractedDriving
//
//  Created by Senthamarai, Ram on 10/14/22.
//

import Foundation
import Vision

/// Project class.
struct Project {
  var model: VNCoreMLModel?
  
  /// Initialize Project instance with MLModel.
  init(mlModel: MLModel?) {
    if let mlModel = mlModel {
      self.model = try? VNCoreMLModel(for: mlModel)
    }
  }
}
