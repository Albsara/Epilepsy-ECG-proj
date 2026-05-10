import 'dart:convert';
import '../models/live_data_model.dart';

class AiEvaluator {
  dynamic _aiModelRoot;

  void loadModel(String jsonString) {
    _aiModelRoot = json.decode(jsonString);
  }

  bool isReady() => _aiModelRoot != null;

  bool evaluate(LiveDataModel data) {
    if (_aiModelRoot == null) return false;

    final input = [
      data.hr.toDouble(),
      data.hrv.toDouble(),
      data.medication.toLowerCase() == 'yes' ? 1.0 : 0.0,
      data.symptoms.toLowerCase() == 'yes' ? 1.0 : 0.0,
      data.sleep.toLowerCase() == 'good' ? 1.0 : 0.0,
      (data.stress.toLowerCase() == 'high' ||
              data.stress.toLowerCase() == 'bad')
          ? 1.0
          : 0.0,
    ];

    final result = _traverseTree(_aiModelRoot, input);
    return result[1] == 1.0;
  }

  List<double> _traverseTree(dynamic node, List<double> input) {
    if (node is List) {
      return [node[0].toDouble(), node[1].toDouble()];
    }

    final int feature = node['f'];
    final double threshold = (node['t'] as num).toDouble();
    if (input[feature] <= threshold) {
      return _traverseTree(node['l'], input);
    } else {
      return _traverseTree(node['r'], input);
    }
  }
}
