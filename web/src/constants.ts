export const number_of_lines_on_a_staff = 5;
export const max_number_of_ledger_lines = 4;

export function tolerance_for_staff_line_detection(unit_size: number): number {
  return unit_size / 3;
}

export function max_line_gap_size(unit_size: number): number {
  return 5 * unit_size;
}

export function is_short_line(unit_size: number): number {
  return unit_size / 5;
}

export function is_short_connected_line(unit_size: number): number {
  return 2 * unit_size;
}

export function min_height_for_brace_rough(unit_size: number): number {
  return 2 * unit_size;
}

export function max_width_for_brace_rough(unit_size: number): number {
  return 3 * unit_size;
}

export function min_height_for_brace(unit_size: number): number {
  return 4 * unit_size;
}

export function tolerance_for_touching_clefs(unit_size: number): number {
  return Math.round(unit_size * 2);
}

export function tolerance_for_staff_at_any_point(_unit_size: number): number {
  return 0;
}

export function tolerance_note_grouping(unit_size: number): number {
  return 1 * unit_size;
}

export function bar_line_max_width(unit_size: number): number {
  return 2 * unit_size;
}

export function bar_line_min_height(unit_size: number): number {
  return 3 * unit_size;
}

export function black_spot_removal_threshold(unit_size: number): number {
  return 2 * unit_size;
}

export const staff_line_segment_x_tolerance = 10;
export const minimum_connections_to_form_combined_staff = 1;
export const duration_of_quarter = 16;
export const image_noise_limit = 50;
export const staff_position_tolerance = 50;
export const max_angle_for_lines_to_be_parallel = 10;
export const NOTEHEAD_SIZE_RATIO = 1.285714;
export const grandstaff_x_distance_threshold_factor = 5;
export const grandstaff_y_overlap_threshold_factor = 0.5;
export const brace_core_width_ratio = 0.5;
export const min_width_for_brace_dot_candidate = 5;
