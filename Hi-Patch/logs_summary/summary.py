import os
import re
import pandas as pd

def extract_experiment_results_newformat(log_folder, output_file):
    results = []
    total_params_dict = {}   # 파일별 total parameters (없으면 Unknown)
    dataset_dict = {}        # 파일별 dataset 이름 (가능하면 기록)

    # 정규식 패턴들
    seed_pat = re.compile(r"--seed\s+(\d+)")
    dataset_pat = re.compile(r"--dataset\s+([A-Za-z0-9_\-]+)")
    total_params_pat = re.compile(r"Total Trainable Parameters:\s*(\d+)")
    # Test 결과: Test - Best epoch, Loss, MSE, RMSE, MAE, MAPE: e, loss, mse, rmse, mae, mape%
    test_pat = re.compile(
        r"Test\s*-\s*Best epoch,\s*Loss,\s*MSE,\s*RMSE,\s*MAE,\s*MAPE:\s*\d+,\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*(-?[\d.]+)%"
    )
    avg_train_time_pat = re.compile(r"Avg Train Time per epoch:\s*([\d.]+)s")
    avg_infer_time_pat = re.compile(r"Avg Inference Time per epoch:\s*([\d.]+)s")
    peak_gpu_train_pat = re.compile(r"Avg Peak GPU Mem \(Train\):\s*([\d.]+)\s*MB")
    peak_gpu_infer_pat = re.compile(r"Peak GPU Mem \(Inference\):\s*([\d.]+)\s*MB")

    for log_file in sorted(os.listdir(log_folder)):
        if not log_file.endswith(".log"):
            continue
        file_path = os.path.join(log_folder, log_file)

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # 파일 전역 메타
        total_params = None
        dataset_in_file = None

        # seed별 현재 블록 결과 임시 저장
        current_seed = None
        # seed -> metrics dict
        seed_results = {}

        def ensure_seed_entry(seed):
            if seed not in seed_results:
                seed_results[seed] = {
                    "Dataset": dataset_in_file,
                    "MSE": None,
                    "MAE": None,
                    "Train_Time": None,
                    "Inference_Time": None,
                    "GPU_Train_MB": None,
                    "GPU_Infer_MB": None,
                }

        for line in lines:
            # total params (있으면)
            m = total_params_pat.search(line)
            if m:
                total_params = int(m.group(1))

            # 커맨드라인에서 dataset/seed 파싱 (블록 시작 신호로 사용)
            md = dataset_pat.search(line)
            if md:
                dataset_in_file = md.group(1)

            ms = seed_pat.search(line)
            if ms:
                s = int(ms.group(1))
                if s in {0, 1, 2, 3, 4}:   # 필요한 seed만
                    current_seed = s
                    ensure_seed_entry(current_seed)
                    # 최신 dataset 반영
                    seed_results[current_seed]["Dataset"] = dataset_in_file
                else:
                    current_seed = None  # 관심 없는 seed

            # Test - Best epoch ... (마지막 등장 값으로 덮어쓰기)
            mt = test_pat.search(line)
            if mt and current_seed in seed_results:
                # loss = float(mt.group(1))  # 필요하면 보관
                mse = float(mt.group(2))
                # rmse = float(mt.group(3))
                mae = float(mt.group(4))
                seed_results[current_seed]["MSE"] = mse
                seed_results[current_seed]["MAE"] = mae

            # 평균 시간/메모리 (블록 말미)
            mt_train = avg_train_time_pat.search(line)
            if mt_train and current_seed in seed_results:
                seed_results[current_seed]["Train_Time"] = float(mt_train.group(1))

            mt_infer = avg_infer_time_pat.search(line)
            if mt_infer and current_seed in seed_results:
                seed_results[current_seed]["Inference_Time"] = float(mt_infer.group(1))

            mgpu_tr = peak_gpu_train_pat.search(line)
            if mgpu_tr and current_seed in seed_results:
                seed_results[current_seed]["GPU_Train_MB"] = float(mgpu_tr.group(1))

            mgpu_inf = peak_gpu_infer_pat.search(line)
            if mgpu_inf and current_seed in seed_results:
                seed_results[current_seed]["GPU_Infer_MB"] = float(mgpu_inf.group(1))

        # 파일 메타 저장
        total_params_dict[log_file] = total_params if total_params is not None else "Unknown"
        dataset_dict[log_file] = dataset_in_file if dataset_in_file is not None else "Unknown"

        # 수집된 seed 결과들을 테이블 rows로 변환 (필수 항목들 존재 시)
        for seed, metrics in seed_results.items():
            # MSE/MAE와 시간 2개는 필수로 체크
            if (
                metrics["MSE"] is not None
                and metrics["MAE"] is not None
                and metrics["Train_Time"] is not None
                and metrics["Inference_Time"] is not None
            ):
                results.append([
                    log_file,
                    metrics.get("Dataset", "Unknown"),
                    seed,
                    metrics["MSE"],
                    metrics["MAE"],
                    metrics["Train_Time"],
                    metrics["Inference_Time"],
                    metrics["GPU_Train_MB"],
                    metrics["GPU_Infer_MB"],
                    total_params_dict[log_file],
                ])

    # 데이터프레임 구성
    df = pd.DataFrame(
        results,
        columns=[
            "Log File",
            "Dataset",
            "Seed",
            "MSE",
            "MAE",
            "Train Time",
            "Inference Time",
            "GPU Train MB",
            "GPU Infer MB",
            "Total Parameters",
        ],
    )

    if df.empty:
        # 아무 것도 못 찾았을 때 방어
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("No results parsed. Check log patterns or folder path.\n")
        return df, None

    # 정렬
    df = df.sort_values(by=["Log File", "Seed"]).reset_index(drop=True)

    # 파일별 요약 출력
    mse_means = {}

    with open(output_file, "w", encoding="utf-8") as f:
        for log_file in df["Log File"].unique():
            file_df = df[df["Log File"] == log_file]
            total_params = total_params_dict.get(log_file, "Unknown")
            dataset_name = dataset_dict.get(log_file, "Unknown")

            f.write(f"\n### {log_file} ###\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Total Trainable Parameters: {total_params}\n")

            # 각 시드별 결과
            for _, row in file_df.iterrows():
                f.write(
                    "Seed {seed}: MSE={mse:.5f}, MAE={mae:.5f}, "
                    "Train Time={tt:.5f}s, Inference Time={it:.5f}s, "
                    "GPU Train={gt} MB, GPU Infer={gi} MB\n".format(
                        seed=int(row["Seed"]),
                        mse=row["MSE"],
                        mae=row["MAE"],
                        tt=row["Train Time"],
                        it=row["Inference Time"],
                        gt=("NA" if pd.isna(row["GPU Train MB"]) else f"{row['GPU Train MB']:.1f}"),
                        gi=("NA" if pd.isna(row["GPU Infer MB"]) else f"{row['GPU Infer MB']:.1f}"),
                    )
                )

            # 평균/표준편차
            mse_mean, mse_std = file_df["MSE"].mean(), file_df["MSE"].std()
            mae_mean, mae_std = file_df["MAE"].mean(), file_df["MAE"].std()
            train_time_mean = file_df["Train Time"].mean()
            infer_time_mean = file_df["Inference Time"].mean()
            gpu_train_mean = file_df["GPU Train MB"].mean()
            gpu_infer_mean = file_df["GPU Infer MB"].mean()

            mse_means[log_file] = mse_mean

            f.write(
                "Mean(Std): MSE={:.5f} ({:.5f}), MAE={:.5f} ({:.5f}), "
                "Train Time={:.5f}s, Inference Time={:.5f}s".format(
                    mse_mean, mse_std if pd.notna(mse_std) else 0.0,
                    mae_mean, mae_std if pd.notna(mae_std) else 0.0,
                    train_time_mean, infer_time_mean
                )
            )
            # GPU 평균은 있을 때만 추가
            if pd.notna(gpu_train_mean) or pd.notna(gpu_infer_mean):
                f.write(", GPU Train={:.1f} MB, GPU Infer={:.1f} MB".format(
                    gpu_train_mean if pd.notna(gpu_train_mean) else float('nan'),
                    gpu_infer_mean if pd.notna(gpu_infer_mean) else float('nan')
                ))
            f.write("\n")

        # 베스트 로그 파일 (MSE 평균 기준 최소)
        best_log_file = min(mse_means, key=mse_means.get)
        best_file_df = df[df["Log File"] == best_log_file]
        best_mse_mean, best_mse_std = best_file_df["MSE"].mean(), best_file_df["MSE"].std()
        best_mae_mean, best_mae_std = best_file_df["MAE"].mean(), best_file_df["MAE"].std()
        best_train_time_mean = best_file_df["Train Time"].mean()
        best_infer_time_mean = best_file_df["Inference Time"].mean()
        best_gpu_train_mean = best_file_df["GPU Train MB"].mean()
        best_gpu_infer_mean = best_file_df["GPU Infer MB"].mean()
        best_dataset = dataset_dict.get(best_log_file, "Unknown")
        best_total_params = total_params_dict.get(best_log_file, "Unknown")

        f.write(f"\n### Best Log File: {best_log_file} ###\n")
        f.write(f"Dataset: {best_dataset}\n")
        f.write(f"Total Trainable Parameters: {best_total_params}\n")
        f.write(
            "Lowest MSE Mean: {:.5f} ({:.5f}), "
            "MAE Mean: {:.5f} ({:.5f}), "
            "Train Time Mean: {:.5f}s, Inference Time Mean: {:.5f}s".format(
                best_mse_mean, best_mse_std if pd.notna(best_mse_std) else 0.0,
                best_mae_mean, best_mae_std if pd.notna(best_mae_std) else 0.0,
                best_train_time_mean, best_infer_time_mean
            )
        )
        if pd.notna(best_gpu_train_mean) or pd.notna(best_gpu_infer_mean):
            f.write(", GPU Train={:.1f} MB, GPU Infer={:.1f} MB".format(
                best_gpu_train_mean if pd.notna(best_gpu_train_mean) else float('nan'),
                best_gpu_infer_mean if pd.notna(best_gpu_infer_mean) else float('nan')
            ))
        f.write("\n")

    return df, best_log_file


# 사용 예시
if __name__ == "__main__":
    log_folder = "logs_summary"  # 실제 로그 폴더
    output_file = "logs_summary/experiment_results.txt"
    df_results, best_log = extract_experiment_results_newformat(log_folder, output_file)
    print(f"Experiment results saved to {output_file}")
    print(f"Log file with lowest MSE mean: {best_log}")
