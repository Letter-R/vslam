use image::{DynamicImage, GenericImageView, ImageBuffer, Rgba, Rgb};
use nalgebra::Vector2;
use std::path::Path;

use vslam_frontend::orb::ORB;
use vslam_core::keyframe::Feature;

fn main(){
    // 读取两幅图像
    let image1 = image::open(Path::new("C:/Users/Administrator/Documents/vslam/data/V1_01_easy/mav0/cam0/data/1403715412862142976.png")).unwrap();
    let image2 = image::open(Path::new("C:/Users/Administrator/Documents/vslam/data/V1_01_easy/mav0/cam0/data/1403715413312143104.png")).unwrap();

    // 提取ORB特征
    let features1 = ORB::extract_features(&image1);
    let features2 = ORB::extract_features(&image2);

    // 匹配特征点
    let matches = ORB::match_features(&features1,& features2);
    println!("{:?}",&matches);

    // 将两幅图像拼接在一起
    let mut combined_image = ImageBuffer::new(image1.width() * 2, image1.height());
    for y in 0..image1.height() {
        for x in 0..image1.width() {
            let pixel1 = image1.get_pixel(x, y);
            let pixel2 = image2.get_pixel(x, y);
            combined_image.put_pixel(x, y, Rgb([pixel1.0[0], pixel1.0[0], pixel1.0[0]]));
            combined_image.put_pixel(x + image1.width(), y, Rgb([pixel2.0[0], pixel2.0[0], pixel2.0[0]]));
        }
    }

    // 在图像上画出特征点
    let draw_feature = |image: &mut ImageBuffer<Rgb<u8>, Vec<u8>>, x: u32, y: u32, color: Rgb<u8>| {
        let radius = 3;
        for dy in -(radius as i32)..=(radius as i32) {
            for dx in -(radius as i32)..=(radius as i32) {
                let xx = (x as i32 + dx) as u32;
                let yy = (y as i32 + dy) as u32;
                if xx < image.width() && yy < image.height() {
                    image.put_pixel(xx, yy, color);
                }
            }
        }
    };
    for feature in &features1 {
        draw_feature(&mut combined_image, feature.location.x as u32, feature.location.y as u32, Rgb([255, 0, 0]));
    }    
    for feature in &features2 {
        draw_feature(&mut combined_image, feature.location.x as u32 + image1.width(), feature.location.y as u32, Rgb([255, 0, 0]));
    }

    // 画出匹配的连线
    let draw_line = |image: &mut ImageBuffer<Rgb<u8>, Vec<u8>>, x1: u32, y1: u32, x2: u32, y2: u32, color: Rgb<u8>| {
        let dx = x2 as f64 - x1 as f64;
        let dy = y2 as f64 - y1 as f64;
        let length = (dx * dx + dy * dy).sqrt();
        let steps = (length * 2.0) as usize;
        for i in 0..=steps {
            let t = i as f64 / steps as f64;
            let x = x1 as f64 + t * dx;
            let y = y1 as f64 + t * dy;
            if x < image.width() as f64 && y < image.height() as f64 {
                image.put_pixel(x as u32, y as u32, color);
            }
        }
    };

    for (i1, i2) in matches {
        let feature1 = &features1[i1];
        let feature2 = &features2[i2];
        draw_line(
            &mut combined_image,
            feature1.location.x as u32,
            feature1.location.y as u32,
            feature2.location.x as u32 + image1.width(),
            feature2.location.y as u32,
            Rgb([0, 255, 0]),
        );
    }

    // 保存可视化结果
    combined_image.save("matched_features.png").unwrap();

}
