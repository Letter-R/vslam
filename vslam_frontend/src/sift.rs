use image::{DynamicImage, imageops::FilterType, GenericImageView, Pixel, ImageBuffer, Luma};
use nalgebra::{Vector2, Vector3, Matrix3};
use vslam_core::keyframe::Feature;
use imageproc::filter::gaussian_blur_f32;

const OCTAVES: usize = 4; // 尺度空间层数
const INTERVALS: usize = 3; // 单个尺度空间的影像层数
const SIGMA: f64 = 1.6; // σ，标准差，高斯算子的参数

#[derive(Clone, Copy)]
pub struct SIFT {
    pub location: Vector2<f64>,
    pub descriptor: [u64; 4],
}

impl Feature for SIFT {
    fn extract_features(image: &DynamicImage) -> Vec<Self>
    where
        Self: Sized,
    {
        let scale_space = build_scale_space(&image);
        let dog_space = build_difference_of_gaussians(&scale_space);
        let keypoints = find_keypoints(&dog_space);
        
        // 在这里添加关键点精确化和方向分配步骤（Step 4）
        
        // 在这里添加描述子计算步骤（Step 5）
        keypoints
    }

    fn match_features(feature1: &Vec<Self>, feature2: &Vec<Self>) -> Vec<(usize, usize)>
    where
        Self: Sized,
    {
        todo!()
    }
}

/// 构建高斯金字塔
/// OCTAVES层
/// 每层INTERVALS + 3张影像
fn build_scale_space(image: &DynamicImage) -> Vec<Vec<DynamicImage>> {
    let mut scale_space =
        vec![vec![DynamicImage::new_luma8(image.width(), image.height()); INTERVALS + 3]; OCTAVES];

    for octave in 0..OCTAVES {
        for interval in 0..(INTERVALS + 3) {
            let sigma = get_sigma(octave, interval);
            let scaled_image = if octave == 0 && interval == 0 {
                image.clone()
            } else if interval == 0 {
                let (width, height) = (
                    scale_space[octave - 1][0].width() / 2,
                    scale_space[octave - 1][0].height() / 2,
                );
                scale_space[octave - 1][0].resize(width, height, FilterType::Triangle)
            } else {
                smooth_image(&scale_space[octave][interval - 1], sigma)
            };
            scale_space[octave][interval] = scaled_image;
        }
    }

    scale_space
}

/// 根据OCTAVES和INTERVAL计算高斯滤波的σ
fn get_sigma(octave: usize, interval: usize) -> f32 {
    let k = 2f32.powf(1.0 / INTERVALS as f32);
    SIGMA as f32 * k.powi(octave as i32 + interval as i32)
}

/// 高斯滤波
fn smooth_image(image: &DynamicImage, sigma: f32) -> DynamicImage {
    let luma_image = image.to_luma8();
    let smooth_image = gaussian_blur_f32(&luma_image, sigma);
    DynamicImage::ImageLuma8(smooth_image)
}

/// 构建高斯差分金字塔
/// OCTAVES层
/// 每层INTERVALS + 2张影像
fn build_difference_of_gaussians(scale_space: &Vec<Vec<DynamicImage>>) -> Vec<Vec<DynamicImage>> {
    let mut dog_space = vec![vec![DynamicImage::new_luma8(scale_space[0][0].width(), scale_space[0][0].height()); INTERVALS+2]; OCTAVES];

    for octave in 0..OCTAVES {
        for interval in 0..INTERVALS+2 {
            let width = scale_space[octave][interval].width();
            let height = scale_space[octave][interval].height();
            let mut difference_image = image::ImageBuffer::new(width, height);
            for y in 0..height {
                for x in 0..width {
                    let value1 = scale_space[octave][interval].get_pixel(x, y)[0] as f32;
                    let value2 = scale_space[octave][interval + 1].get_pixel(x, y)[0] as f32;
                    difference_image.put_pixel(x, y, image::Luma([((value1 - value2) + 128.0) as u8]));
                }
            }
            dog_space[octave][interval] = DynamicImage::ImageLuma8(difference_image);
        }
    }

    dog_space
}

/// 检测特征点
/// 每组检测INTERVAL个尺度特征点
fn find_keypoints(dog_space: &Vec<Vec<DynamicImage>>) -> Vec<SIFT> {
    let mut keypoints = Vec::new();

    for octave in 0..OCTAVES {
        for interval in 1..(INTERVALS +1) {
            for y in 1..(dog_space[octave][interval].height() - 1) {
                for x in 1..(dog_space[octave][interval].width() - 1) {
                    if is_extrema(&dog_space[octave], interval, x, y) {
                        
                        if let Some(location) = refine_keypoint(&dog_space, octave, interval, x, y) {
                            let descriptor = [0u64; 4];
                            keypoints.push(SIFT { location, descriptor });
                        }

                    }
                }
            }
        }
    }

    keypoints
}

/// 判断特征点，三维26个
fn is_extrema(dog_space: &Vec<DynamicImage>, interval: usize, x: u32, y: u32) -> bool {
    let center_pixel = dog_space[interval].get_pixel(x, y).to_luma()[0];

    for i in (interval - 1)..=(interval + 1) {
        for j in (y - 1)..=(y + 1) {
            for k in (x - 1)..=(x + 1) {
                let neighbor_pixel = dog_space[i].get_pixel(k, j).to_luma()[0];
                if i == interval && j == y && k == x {
                    continue;
                }
                if center_pixel <= neighbor_pixel {
                    return false;
                }
            }
        }
    }

    true
}

/// 精确化精确点
fn refine_keypoint(dog_space: &Vec<Vec<DynamicImage>>, octave: usize, interval: usize, x: u32, y: u32) -> Option<Vector2<f64>> {
    const MAX_ITERATIONS: u8 = 5; // 定义迭代次数上限
    const CONTRAST_THRESHOLD: f64 = 0.03;// 定义关键点对比度阈值
    const EDGE_THRESHOLD: f64 = 10.0;// 定义边缘响应阈值
    
    let mut x = x as f64;
    let mut y = y as f64;
    let mut interval = interval as f64;
    
    for _ in 0..MAX_ITERATIONS {
        let gradients = compute_gradients(&dog_space, octave, interval as usize, x, y);
        let hessian = compute_hessian(&dog_space, octave, interval as usize, x, y);
        let offset = -hessian.try_inverse().unwrap() * gradients;
        
        if offset.norm() <= 0.5 {
            let contrast = dog_space[octave][interval as usize].get_pixel(x as u32, y as u32).to_luma()[0] as f64 / 255.0 + 0.5 * gradients.dot(&offset);
            if contrast.abs() < CONTRAST_THRESHOLD {
                return None;
            }

            let trace = hessian[(0, 0)] + hessian[(1, 1)];
            let determinant = hessian[(0, 0)] * hessian[(1, 1)] - hessian[(0, 1)] * hessian[(1, 0)];
            let curvature_ratio = trace * trace / determinant;

            if determinant <= 0.0 || curvature_ratio > (EDGE_THRESHOLD + 1.0) * (EDGE_THRESHOLD + 1.0) / EDGE_THRESHOLD {
                return None;
            }

            break;
        }
        
        x += offset[0];
        y += offset[1];
        interval += offset[2];
        
        if x < 1.0 || x > dog_space[octave][0].width() as f64 - 2.0 || y < 1.0 || y > dog_space[octave][0].height() as f64 - 2.0 || interval < 1.0 || interval > INTERVALS as f64 {
            return None;
        }
    }
    

    
    Some(Vector2::new(x, y))
}

/// 计算图像的梯度幅值和方向
fn compute_gradients(dog_space: &Vec<Vec<DynamicImage>>, octave: usize, interval: usize, x: f64, y: f64) -> Vector3<f64> {
    let dx = (dog_space[octave][interval].get_pixel((x + 1.0) as u32, y as u32).to_luma()[0] as f64
        - dog_space[octave][interval].get_pixel((x - 1.0) as u32, y as u32).to_luma()[0] as f64) / 2.0;
    let dy = (dog_space[octave][interval].get_pixel(x as u32, (y + 1.0) as u32).to_luma()[0] as f64
        - dog_space[octave][interval].get_pixel(x as u32, (y - 1.0) as u32).to_luma()[0] as f64) / 2.0;
    let ds = (dog_space[octave][interval + 1].get_pixel(x as u32, y as u32).to_luma()[0] as f64
        - dog_space[octave][interval - 1].get_pixel(x as u32, y as u32).to_luma()[0] as f64) / 2.0;

    Vector3::new(dx, dy, ds)
}

/// 计算海森矩阵
/// 一个3x3矩阵
/// 关键点位置处的二阶导数
fn compute_hessian(dog_space: &Vec<Vec<DynamicImage>>, octave: usize, interval: usize, x: f64, y: f64) -> Matrix3<f64> {
    let dxx = dog_space[octave][interval].get_pixel((x + 1.0) as u32, y as u32).to_luma()[0] as f64
        - 2.0 * dog_space[octave][interval].get_pixel(x as u32, y as u32).to_luma()[0] as f64
        + dog_space[octave][interval].get_pixel((x - 1.0) as u32, y as u32).to_luma()[0] as f64;

    let dyy = dog_space[octave][interval].get_pixel(x as u32, (y + 1.0) as u32).to_luma()[0] as f64
        - 2.0 * dog_space[octave][interval].get_pixel(x as u32, y as u32).to_luma()[0] as f64
        + dog_space[octave][interval].get_pixel(x as u32, (y - 1.0) as u32).to_luma()[0] as f64;

    let dss = dog_space[octave][interval + 1].get_pixel(x as u32, y as u32).to_luma()[0] as f64
        - 2.0 * dog_space[octave][interval].get_pixel(x as u32, y as u32).to_luma()[0] as f64
        + dog_space[octave][interval - 1].get_pixel(x as u32, y as u32).to_luma()[0] as f64;

    let dxy = ((dog_space[octave][interval].get_pixel((x + 1.0) as u32, (y + 1.0) as u32).to_luma()[0] as f64
        - dog_space[octave][interval].get_pixel((x - 1.0) as u32, (y + 1.0) as u32).to_luma()[0] as f64)
        - (dog_space[octave][interval].get_pixel((x + 1.0) as u32, (y - 1.0) as u32).to_luma()[0] as f64
        - dog_space[octave][interval].get_pixel((x - 1.0) as u32, (y - 1.0) as u32).to_luma()[0] as f64)) / 4.0;

    let dxs = ((dog_space[octave][interval + 1].get_pixel((x + 1.0) as u32, y as u32).to_luma()[0] as f64
        - dog_space[octave][interval - 1].get_pixel((x + 1.0) as u32, y as u32).to_luma()[0] as f64)
        - (dog_space[octave][interval + 1].get_pixel((x - 1.0) as u32, y as u32).to_luma()[0] as f64
        - dog_space[octave][interval - 1].get_pixel((x - 1.0) as u32, y as u32).to_luma()[0] as f64)) / 4.0;
    let dys = ((dog_space[octave][interval + 1].get_pixel(x as u32, (y + 1.0) as u32).to_luma()[0] as f64
        - dog_space[octave][interval - 1].get_pixel(x as u32, (y + 1.0) as u32).to_luma()[0] as f64)
        - (dog_space[octave][interval + 1].get_pixel(x as u32, (y - 1.0) as u32).to_luma()[0] as f64
        - dog_space[octave][interval - 1].get_pixel(x as u32, (y - 1.0) as u32).to_luma()[0] as f64)) / 4.0;
    
    Matrix3::new(dxx, dxy, dxs,
                 dxy, dyy, dys,
                 dxs, dys, dss)
    }