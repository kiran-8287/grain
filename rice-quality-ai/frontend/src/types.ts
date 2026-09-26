export interface DefectItem {
  broken_label?: string;
  broken_ratio?: number;
  is_small_broken?: boolean;
  damaged_label?: string;
  damaged_probability?: number;
  discoloured_label?: string;
  colour_distance?: number;
  discoloured_confidence?: number;
  chalky_label?: string;
  chalky_probability?: number;
  red_label?: string;
  red_area_fraction?: number;
  red_confidence?: number;
  red_method?: string;
  dehusked_label?: string;
  dehusked_surface_fraction?: number;
  dehusked_confidence?: number;
  immature_shrunken_status?: string;
  proxy_score?: number;
  sprouted_weevilled_label?: string;
  probability?: number;
  confidence?: number;
  confidence_basis?: string;
  predicted_class_index?: number;
  class_mapping?: Record<string, string>;
  method: string;
  limitation?: string;
  reason?: string;
  _source?: string;
  [key: string]: any;
}

export interface GrainGeometry {
  grain_id: number;
  area_pixels: number;
  perimeter_pixels: number;
  centroid: [number, number];
  orientation_deg: number;
  bbox: [number, number, number, number];
  major_axis_pixels: number;
  minor_axis_pixels: number;
  solidity: number;
  aspect_ratio: number;
  length_pixels: number;
  breadth_pixels: number;
  length_mm?: number | null;
  breadth_mm?: number | null;
  lb_ratio?: number | null;
  measurement_quality: string;
  is_anomalous?: boolean;
  anomaly_reasons?: string[];
}

export interface GrainInstance {
  id: number;
  bbox: [number, number, number, number];
  centroid: [number, number];
  confidence: number;
  segmentation_quality: string;
  is_touching: boolean;
  contour?: number[][];
  geometry: GrainGeometry;
  defects: {
    broken: DefectItem;
    damaged: DefectItem;
    discoloured: DefectItem;
    chalky: DefectItem;
    red: DefectItem;
    dehusked: DefectItem;
    immature_shrunken: DefectItem;
    sprouted_weevilled: DefectItem;
  };
  sample_level_parameters: {
    foreign_matter: string;
    admixture: string;
    total_count: string;
  };
}

export interface ForeignObject {
  foreign_id: number;
  class: string;
  confidence: number;
  bbox: [number, number, number, number];
  area_pixels?: number;
}

export interface SampleSummary {
  total_rice_grains: number;
  uncertain_grains: number;
  rejected_grains: number;
  foreign_matter_count: number;
  admixture_percentage: number | null;
  broken_count: number;
  broken_percent: number | null;
  broken_analyzed_count?: number;
  damaged_count: number;
  damaged_percent: number | null;
  damaged_analyzed_count?: number;
  discoloured_count: number;
  discoloured_percent: number | null;
  discoloured_analyzed_count?: number;
  chalky_count: number;
  chalky_percent: number | null;
  chalky_analyzed_count?: number;
  red_count: number;
  red_percent: number | null;
  red_analyzed_count?: number;
  dehusked_count: number;
  dehusked_percent: number | null;
  dehusked_analyzed_count?: number;
  immature_count: number;
  immature_percent: number | null;
  immature_analyzed_count?: number;
  sprouted_count: number;
  sprouted_percent: number | null;
  sprouted_analyzed_count?: number;
  admixture_analyzed_count?: number;
  foreign_matter_analyzed_count?: number;
  average_length: number;
  median_length: number;
  average_breadth: number;
  median_breadth: number;
  average_lb_ratio: number;
  median_lb_ratio: number;
  measurement_unit: string;
  calibration_mode: string;
  pixels_per_mm?: number | null;
  is_small_sample: boolean;
  label_note: string;
}

export interface ScreeningParameter {
  observed_value: string;
  reference_limit: string;
  status: 'WITHIN REFERENCE LIMIT' | 'EXCEEDS REFERENCE LIMIT' | 'NOT ASSESSABLE' | 'NOT DETERMINABLE' | 'NOT APPLICABLE';
  basis: string;
  detected_count?: number;
  analyzed_count?: number;
  observed_fraction?: number | null;
  difference?: string;
  notes?: string;
}

export interface StandardsResult {
  standard_name?: string;
  season?: string;
  grade_profile?: string;
  screening: Record<string, ScreeningParameter>;
  standard_reference?: {
    source?: string;
    season?: string;
    display_label?: string;
    verification_status?: string;
    grade_profile?: string;
  };
  official_grade: {
    status: string;
    reason: string;
    disclaimer: string;
    message?: string;
    reasons?: string[];
    laboratory_requirements_missing?: string[];
  };
  warnings?: string[];
}

export interface RiceGateDetection {
  object_id: number;
  class_id: number;
  class_name: string;
  is_rice: boolean;
  confidence: number;
  confidence_threshold?: number;
  bbox: [number, number, number, number];
  area_pixels: number;
  rice_likeness_cues?: Record<string, number>;
  features?: Record<string, number | number[]>;
}

export interface RiceGate {
  status: 'RICE' | 'NOT_RICE' | 'NOT_RICE_SPARSE' | 'NO_ANALYSABLE_RICE' | 'NO_RICE_CLASS';
  has_rice: boolean;
  analysis_stopped: boolean;
  message: string;
  reason?: string;
  rice_class_name: string | null;
  rice_class_id: number | null;
  foreign_matter_classes: string[];
  rice_confidence_threshold: number;
  total_detections: number;
  rice_detections: number;
  foreign_matter_detections: number;
  rice_fraction?: number;
  min_rice_fraction_required?: number | null;
  rice_confidence_mean: number;
  rice_confidence_max: number;
  detections: RiceGateDetection[];
  class_mapping: {
    segmentation: Record<string, number>;
    foreign_matter: Record<string, number>;
  };
  class_mapping_source?: Record<string, string>;
  method: string;
  model_status?: string;
  limitation?: string;
  debug: string;
  warnings: string[];
}

export interface ImageQuality {
  tier: 'GOOD' | 'FAIR' | 'POOR' | 'UNRELIABLE';
  blur_score: number;
  median_grain_pixels: number;
  mean_segmentation_confidence: number;
  reasons: string[];
}

export interface AnalysisResult {
  success: boolean;
  job_id?: string;
  rice_detected: boolean;
  message?: string;
  rice_gate?: RiceGate;
  image: {
    width: number;
    height: number;
    channels: number;
    megapixels: number;
    format: string;
    mode: string;
  };
  sample: {
    total_detected: number;
    analysed: number;
    uncertain: number;
    rejected: number;
    estimated_merged?: number;
  };
  calibration: {
    calibrated: boolean;
    mode: string;
    pixels_per_mm?: number | null;
    marker_count?: number;
    message?: string;
  };
  quality?: ImageQuality;
  grains: GrainInstance[];
  foreign_matter: ForeignObject[];
  foreign_matter_summary?: {
    foreign_object_count: number;
    foreign_matter_fraction_estimate: number;
    total_image_area: number;
    method: string;
    basis: string;
    warnings?: string[];
    [key: string]: any;
  };
  admixture?: {
    admixture_status: string;
    admixture_count: number;
    admixture_percentage: number;
    admixture_confidence: number;
    method: string;
  };
  summary: SampleSummary;
  standards: StandardsResult;
  annotated_image_base64?: string;
  warnings: string[];
  processing_time_seconds: number;
}
