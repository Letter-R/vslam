use image::{DynamicImage, GrayImage};
use nalgebra::Vector2;
use rand::Rng;
use vslam_core::keyframe::Feature;

#[derive(Clone, Copy)]
pub struct ORB{
    pub location:Vector2<f64>,
    pub descriptor:[u64;4],
}

impl Feature for ORB {
    fn extract_features(image:&DynamicImage)->Vec<Self> where Self:Sized {
        let gray_image=image.to_luma8();
        let key_points=fast(&gray_image);
        let descriptors=brief(&gray_image,&key_points);
        key_points.into_iter().zip(descriptors).map(|(location, descriptor)|{ORB { location, descriptor}})
        .collect()
    }

    fn match_features(feature1:&Vec<Self>,feature2:&Vec<Self>)->Vec<(usize,usize)> where Self:Sized {
        let ratio_threshold = 0.8;
        let mut matches = Vec::new();

        for (i, feature1_descriptor) in feature1.iter().enumerate() {
            let mut best_distance = std::u32::MAX;
            let mut second_best_distance = std::u32::MAX;
            let mut best_index = None;

            // 保留最小的两个
            for (j, feature2_descriptor) in feature2.iter().enumerate() {
                let distance = hamming_distance(&feature1_descriptor.descriptor, &feature2_descriptor.descriptor);
                if distance < best_distance {
                    second_best_distance = best_distance;
                    best_distance = distance;
                    best_index = Some(j);
                } else if distance < second_best_distance {
                    second_best_distance = distance;
                }
            }
            
            // 比值测试
            if best_distance < (ratio_threshold * second_best_distance as f64) as u32 {
                if let Some(best_j) = best_index {
                    matches.push((i, best_j));
                }
            }
        }

        matches
    }

}

/// fast角点检测
fn fast(image: &GrayImage) -> Vec<Vector2<f64>> {
    let threshold:u8=20;
    let border=3;
    let width = image.width();
    let height=image.height();
    let mut keypoints:Vec<Vector2<f64>>=Vec::new();

    // 通过is_corner_fast检测的，加入keypoints中
    for y in border..(height-border){
        for x in border..(width-border){
            let pixel_value=image.get_pixel(x,y).0[0];
            if is_corner_fast(image, x, y, pixel_value, threshold) {
                keypoints.push(Vector2::new(x as f64, y as f64));
            }
        }
    }

    keypoints
}

/// 判断输入是否是fast角点
fn is_corner_fast(image: &GrayImage, x: u32, y: u32, pixel_value: u8, threshold: u8) -> bool {
    let offsets: [(i32, i32); 16] = [(0, 3),(1, 3),        (2, 2),        (3, 1),        (3, 0),        (3, -1),        (2, -2),        (1, -3),        (0, -3),        (-1, -3),        (-2, -2),        (-3, -1),        (-3, 0),        (-3, 1),        (-2, 2),        (-1, 3),    ];

    let darker = |value| (value < pixel_value) && ( pixel_value -value > threshold);
    let brighter = |value| (value > pixel_value) && (value - pixel_value > threshold);

    let darker_brighter:Vec<(bool,bool)>= offsets
        .iter()
        .map(|&(dx, dy)| {
            let pixel = image.get_pixel((x as i32 + dx) as u32, (y as i32 + dy) as u32).0[0];
            (darker(pixel), brighter(pixel))
        })
        .collect::<Vec<_>>();

    let consecutive_darker_or_brighter = |count: usize| -> bool {
        let mut consecutive_darker = 0;
        let mut consecutive_brighter = 0;
        for &(darker, brighter) in darker_brighter.iter().cycle().take(darker_brighter.len() * 2) {
            if darker {
                consecutive_darker += 1;
                consecutive_brighter = 0;
            } else if brighter {
                consecutive_brighter += 1;
                consecutive_darker = 0;
            } else {
                consecutive_darker = 0;
                consecutive_brighter = 0;
            }
            if consecutive_darker >= count || consecutive_brighter >= count {
                return true;
            }
        }
        false
    };


    consecutive_darker_or_brighter(9) 
}

/// brief描述子
fn brief(image: &GrayImage, keypoints: &[Vector2<f64>]) -> Vec<[u64; 4]> {
    let patch_size = 31;
    let border = patch_size / 2;
    let width = image.width();
    let height = image.height();
    let mut rng = rand::thread_rng();
    let random_points = (0..256)
        .map(|_| {
            (
                rng.gen_range(-border..=border),
                rng.gen_range(-border..=border),
            )
        })
        .collect::<Vec<_>>();

    keypoints
        .iter()
        .map(|&keypoint| {
            let x = keypoint[0] as i32;
            let y = keypoint[1] as i32;

            if x - border < 0
                || x + border >= width as i32
                || y - border < 0
                || y + border >= height as i32
            {
                return [0; 4];
            }

            let mut descriptor = [0u64; 4];
            for (i, &(dx, dy)) in random_points.iter().enumerate() {
                let pos1 = (x + dx, y + dy);
                let pos2 = (x - dx, y - dy);
                let intensity1 = image.get_pixel(pos1.0 as u32, pos1.1 as u32).0[0];
                let intensity2 = image.get_pixel(pos2.0 as u32, pos2.1 as u32).0[0];
                if intensity1 < intensity2 {
                    let idx = i / 64;
                    let bit = i % 64;
                    descriptor[idx] |= 1u64 << bit;
                }
            }
            descriptor
        })
        .collect()
}

/// 计算汉明距离
fn hamming_distance(a: &[u64; 4], b: &[u64; 4]) -> u32 {
    let mut distance = 0;
    for i in 0..4 {
        distance += (a[i] ^ b[i]).count_ones();
    }
    distance
}