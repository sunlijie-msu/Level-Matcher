# =============================================================================
# Run_Pipeline.ps1 — Full Level Matcher Workflow Launcher
# =============================================================================
#
# High-level Structure and Workflow Explanation
# ---------------------------------------------
# This script launches the full Level Matcher pipeline in a new external
# PowerShell window, completely detached from VS Code. This prevents VS Code
# from freezing during the computationally intensive Level_Matcher.py step.
#
# Workflow:
#
#   [Run_Pipeline.ps1 in VS Code terminal]
#           |
#           | Start-Process (detached, new window)
#           v
#   [External PowerShell Window]
#           |
#           |-- Step 1: Dataset_Parser.py    (parses ENSDF -> JSON)
#           |-- Step 2: Level_Matcher.py     (XGBoost training + inference)
#           |-- Step 3: Combined_Visualizer.py (generates figures)
#           |
#           v
#   Outputs: outputs/pairwise/, outputs/clustering/, outputs/figures/
#
# Usage:
#   From VS Code terminal:  .\.github\temp\Run_Pipeline.ps1
#   From Windows Explorer:  Double-click this file (right-click -> Run with PowerShell)
#   From external terminal: .\Run_Pipeline.ps1
#
# =============================================================================

$python  = "C:\Users\sun\AppData\Local\Programs\Python\Python311\python.exe"
$project = "d:\X\ND\Level-Matcher"

# Detect if we are already running in the external window (flag argument) or
# if we are being launched from VS Code and need to re-spawn externally.
if ($args -contains "--external") {

    # -------------------------------------------------------------------------
    # This block runs INSIDE the detached external window.
    # -------------------------------------------------------------------------
    Set-Location $project

    Write-Host "=============================================" -ForegroundColor Cyan
    Write-Host "  Level Matcher Pipeline — K vs L Datasets  " -ForegroundColor Cyan
    Write-Host "=============================================" -ForegroundColor Cyan
    Write-Host ""

    # Step 1: Dataset_Parser (fast, parses ENSDF files to JSON)
    Write-Host "[1/3] Dataset_Parser.py ..." -ForegroundColor Yellow
    & $python "$project\Dataset_Parser.py"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: Dataset_Parser.py exited with code $LASTEXITCODE" -ForegroundColor Red
        Read-Host "Press Enter to close"
        exit 1
    }
    Write-Host "      Done." -ForegroundColor Green
    Write-Host ""

    # Step 2: Level_Matcher (heavy XGBoost training + pairwise inference)
    Write-Host "[2/3] Level_Matcher.py (XGBoost training — may take several minutes) ..." -ForegroundColor Yellow
    & $python "$project\Level_Matcher.py"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: Level_Matcher.py exited with code $LASTEXITCODE" -ForegroundColor Red
        Read-Host "Press Enter to close"
        exit 1
    }
    Write-Host "      Done." -ForegroundColor Green
    Write-Host ""

    # Step 3: Combined_Visualizer (fast, generates figures from outputs)
    Write-Host "[3/3] Combined_Visualizer.py ..." -ForegroundColor Yellow
    & $python "$project\Combined_Visualizer.py"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: Combined_Visualizer.py exited with code $LASTEXITCODE" -ForegroundColor Red
        Read-Host "Press Enter to close"
        exit 1
    }
    Write-Host "      Done." -ForegroundColor Green
    Write-Host ""

    Write-Host "=============================================" -ForegroundColor Cyan
    Write-Host "  Pipeline Complete                         " -ForegroundColor Cyan
    Write-Host "=============================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Output files:" -ForegroundColor White
    Write-Host "  $project\outputs\pairwise\Output_Level_Pairwise_Inference.txt"
    Write-Host "  $project\outputs\clustering\Output_Clustering_Results_XGBoost.txt"
    Write-Host "  $project\outputs\figures\Input_Level_Scheme.png"
    Write-Host "  $project\outputs\figures\Output_Cluster_Scheme_XGBoost.png"
    Write-Host ""
    Read-Host "Press Enter to close"

} else {

    # -------------------------------------------------------------------------
    # This block runs when triggered from VS Code (or double-clicked).
    # It re-launches this same script in a new detached PowerShell window.
    # VS Code's terminal returns immediately — no freeze.
    # -------------------------------------------------------------------------
    $script_path = $MyInvocation.MyCommand.Path
    Write-Host "Launching pipeline in external window (VS Code will not freeze)..." -ForegroundColor Green
    Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", "`"$script_path`"", "--external"
    Write-Host "External window launched. Monitor progress there." -ForegroundColor Green
}
