//
//  Item.swift
//  remove-background
//
//  Created by normal on 2024/8/21.
//

import Foundation
import SwiftData

@Model
final class Item {
    var timestamp: Date
    
    init(timestamp: Date) {
        self.timestamp = timestamp
    }
}
