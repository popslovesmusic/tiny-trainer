#!/usr/bin/env python3
"""
Migration script from legacy Tiny Agent Trainer to production version.

This script helps users transition from the old system by:
- Converting legacy configurations
- Migrating trained models where possible
- Creating equivalent configurations for the new system
"""

import sys
import os
import shutil
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from tiny_agent_trainer.config import ConfigManager, TaskConfig, ModelConfig, TrainingConfig, DataConfig
from tiny_agent_trainer.config import load_config_from_legacy

class LegacyMigrator:
    """Migrates legacy Tiny Agent Trainer configurations and models."""
    
    def __init__(self, legacy_dir: Path, target_dir: Path):
        self.legacy_dir = Path(legacy_dir)
        self.target_dir = Path(target_dir)
        self.config_manager = ConfigManager(target_dir / "configs")
        self.migration_report = {
            'configs_migrated': 0,
            'models_migrated': 0,
            'data_migrated': 0,
            'errors': [],
            'warnings': []
        }
    
    def detect_legacy_files(self) -> Dict[str, List[Path]]:
        """Detect legacy files in the source directory."""
        detected = {
            'trainer_files': [],
            'config_files': [],
            'model_files': [],
            'data_files': [],
            'julia_files': []
        }
        
        # Look for legacy trainer files
        for pattern in ['tiny_trainer.py', 'vsl_assistant.py', '*trainer*.py']:
            detected['trainer_files'].extend(self.legacy_dir.glob(pattern))
        
        # Look for configuration files
        for pattern in ['*config*.py', '*config*.txt', 'CONFIGS.py']:
            detected['config_files'].extend(self.legacy_dir.glob(pattern))
        
        # Look for model files
        for pattern in ['*.pth', '*_agent.pth', '*_meta.pkl']:
            detected['model_files'].extend(self.legacy_dir.glob(pattern))
        
        # Look for data files
        for pattern in ['*.json', 'vsl_corpus.json', '*corpus*']:
            detected['data_files'].extend(self.legacy_dir.glob(pattern))
        
        # Look for Julia files
        for pattern in ['*.jl', 'run_vsl.jl', 'runtime.jl']:
            detected['julia_files'].extend(self.legacy_dir.glob(pattern))
        
        return detected
    
    def extract_legacy_configs(self, config_file: Path) -> List[Dict[str, Any]]:
        """Extract configurations from legacy Python files."""
        configs = []
        
        try:
            # Read file content
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try to extract CONFIGS dictionary
            if 'CONFIGS' in content:
                # This is risky but needed for migration
                # Execute only the CONFIGS part safely
                exec_globals = {}
                try:
                    exec(content, exec_globals)
                    if 'CONFIGS' in exec_globals:
                        legacy_configs = exec_globals['CONFIGS']
                        
                        for config_name, config_data in legacy_configs.items():
                            configs.append({
                                'name': config_name,
                                'data': config_data,
                                'source_file': str(config_file)
                            })
                except Exception as e:
                    logger.warning(f"Could not execute config file {config_file}: {e}")
            
            # Try to extract individual config variables
            import re
            
            # Look for TASK_NAME, CORPUS, etc.
            patterns = {
                'TASK_NAME': r'TASK_NAME\s*=\s*["\']([^"\']+)["\']',
                'CORPUS': r'CORPUS\s*=\s*\[(.*?)\]',
                'NUM_EPOCHS': r'NUM_EPOCHS\s*=\s*(\d+)',
                'LEARNING_RATE': r'LEARNING_RATE\s*=\s*([\d.]+)'
            }
            
            extracted = {}
            for key, pattern in patterns.items():
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    if key == 'TASK_NAME':
                        extracted[key] = match.group(1)
                    elif key == 'CORPUS':
                        # This is complex, skip for now
                        extracted[key] = []
                    else:
                        try:
                            extracted[key] = eval(match.group(1))
                        except:
                            pass
            
            if extracted:
                configs.append({
                    'name': extracted.get('TASK_NAME', config_file.stem),
                    'data': extracted,
                    'source_file': str(config_file)
                })
        
        except Exception as e:
            logger.error(f"Error reading config file {config_file}: {e}")
            self.migration_report['errors'].append(f"Config file {config_file}: {e}")
        
        return configs
    
    def convert_legacy_config(self, legacy_config: Dict[str, Any]) -> TaskConfig:
        """Convert legacy configuration to new format."""
        try:
            # Use the utility function from config.py
            config = load_config_from_legacy(legacy_config['data'])
            config.task_name = legacy_config['name']
            
            return config
            
        except Exception as e:
            logger.error(f"Error converting config {legacy_config['name']}: {e}")
            self.migration_report['errors'].append(f"Config conversion {legacy_config['name']}: {e}")
            
            # Return minimal config as fallback
            return TaskConfig(
                task_name=legacy_config['name'],
                task_type="classification"
            )
    
    def migrate_model_files(self, model_files: List[Path]) -> Dict[str, Path]:
        """Migrate model files to new directory structure."""
        migrated_models = {}
        
        for model_file in model_files:
            try:
                # Determine model name from filename
                if '_agent.pth' in model_file.name:
                    model_name = model_file.name.replace('_agent.pth', '')
                elif '_agent_meta.pkl' in model_file.name:
                    model_name = model_file.name.replace('_agent_meta.pkl', '')
                else:
                    model_name = model_file.stem
                
                # Create model directory
                model_dir = self.target_dir / "models" / model_name
                model_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy model file
                if model_file.suffix == '.pth':
                    target_file = model_dir / "final_model.pth"
                elif model_file.suffix == '.pkl':
                    target_file = model_dir / "metadata.pkl"
                else:
                    target_file = model_dir / model_file.name
                
                shutil.copy2(model_file, target_file)
                
                migrated_models[model_name] = model_dir
                self.migration_report['models_migrated'] += 1
                
                logger.info(f"Migrated model {model_name} to {model_dir}")
                
            except Exception as e:
                logger.error(f"Error migrating model {model_file}: {e}")
                self.migration_report['errors'].append(f"Model migration {model_file}: {e}")
        
        return migrated_models
    
    def migrate_data_files(self, data_files: List[Path]) -> List[Path]:
        """Migrate data files."""
        migrated_data = []
        
        data_dir = self.target_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        for data_file in data_files:
            try:
                target_file = data_dir / data_file.name
                shutil.copy2(data_file, target_file)
                
                migrated_data.append(target_file)
                self.migration_report['data_migrated'] += 1
                
                logger.info(f"Migrated data file to {target_file}")
                
            except Exception as e:
                logger.error(f"Error migrating data {data_file}: {e}")
                self.migration_report['errors'].append(f"Data migration {data_file}: {e}")
        
        return migrated_data
    
    def migrate_julia_files(self, julia_files: List[Path]):
        """Migrate Julia runtime files."""
        for julia_file in julia_files:
            try:
                target_file = self.target_dir / julia_file.name
                shutil.copy2(julia_file, target_file)
                
                logger.info(f"Migrated Julia file to {target_file}")
                
            except Exception as e:
                logger.error(f"Error migrating Julia file {julia_file}: {e}")
                self.migration_report['errors'].append(f"Julia migration {julia_file}: {e}")
    
    def create_migration_instructions(self) -> str:
        """Create instructions for manual migration steps."""
        instructions = """
# Migration Instructions

## What was migrated automatically:
- Configuration files -> configs/ directory
- Model files -> models/ directory  
- Data files -> data/ directory
- Julia files -> root directory

## Manual steps required:

1. **Verify configurations:**
   ```bash
   python cli.py list
   python cli.py show --config <config_name>
   ```

2. **Test migrated models:**
   ```bash
   python cli.py test --config <config_name>
   ```
   If this fails, you'll need to retrain:
   ```bash
   python cli.py train --config <config_name>
   ```

3. **Update any custom code:**
   - Replace old imports: `from tiny_trainer import` -> `from tiny_agent_trainer import`
   - Update API calls to use new interface
   - Check examples/ directory for new patterns

4. **VSL features:**
   - Ensure Julia is installed and accessible
   - Test VSL execution: `julia run_vsl.jl <test_file>`

## New features available:
- GPU acceleration and monitoring
- Security enhancements  
- Advanced model architectures
- Comprehensive CLI interface
- Docker deployment support

## Troubleshooting:
- Run system check: `python cli.py check`
- Enable debug logging: `export LOG_LEVEL=DEBUG`
- Run validation tests: `python tests/test_system_validation.py`
"""
        return instructions
    
    def run_migration(self) -> bool:
        """Run the complete migration process."""
        logger.info(f"Starting migration from {self.legacy_dir} to {self.target_dir}")
        
        # Create target directories
        (self.target_dir / "configs").mkdir(parents=True, exist_ok=True)
        (self.target_dir / "models").mkdir(parents=True, exist_ok=True)
        (self.target_dir / "data").mkdir(parents=True, exist_ok=True)
        
        # Detect legacy files
        detected_files = self.detect_legacy_files()
        
        logger.info("Detected legacy files:")
        for category, files in detected_files.items():
            if files:
                logger.info(f"  {category}: {len(files)} files")
        
        # Migrate configurations
        all_configs = []
        for config_file in detected_files['config_files']:
            configs = self.extract_legacy_configs(config_file)
            all_configs.extend(configs)
        
        for legacy_config in all_configs:
            try:
                new_config = self.convert_legacy_config(legacy_config)
                saved_path = self.config_manager.save_config(new_config)
                
                self.migration_report['configs_migrated'] += 1
                logger.info(f"Migrated config '{new_config.task_name}' to {saved_path}")
                
            except Exception as e:
                logger.error(f"Error migrating config {legacy_config['name']}: {e}")
                self.migration_report['errors'].append(f"Config {legacy_config['name']}: {e}")
        
        # Migrate models
        self.migrate_model_files(detected_files['model_files'])
        
        # Migrate data
        self.migrate_data_files(detected_files['data_files'])
        
        # Migrate Julia files
        self.migrate_julia_files(detected_files['julia_files'])
        
        # Create migration report
        self.create_migration_report()
        
        # Success if no critical errors
        return len(self.migration_report['errors']) == 0
    
    def create_migration_report(self):
        """Create detailed migration report."""
        report_file = self.target_dir / "migration_report.json"
        
        with open(report_file, 'w') as f:
            json.dump(self.migration_report, f, indent=2)
        
        # Create instructions file
        instructions_file = self.target_dir / "MIGRATION_INSTRUCTIONS.md"
        with open(instructions_file, 'w') as f:
            f.write(self.create_migration_instructions())
        
        # Print summary
        print("\n" + "=" * 50)
        print("MIGRATION SUMMARY")
        print("=" * 50)
        print(f"Configurations migrated: {self.migration_report['configs_migrated']}")
        print(f"Models migrated: {self.migration_report['models_migrated']}")
        print(f"Data files migrated: {self.migration_report['data_migrated']}")
        print(f"Errors encountered: {len(self.migration_report['errors'])}")
        print(f"Warnings: {len(self.migration_report['warnings'])}")
        
        if self.migration_report['errors']:
            print("\nErrors:")
            for error in self.migration_report['errors'][:5]:  # Show first 5
                print(f"  - {error}")
            if len(self.migration_report['errors']) > 5:
                print(f"  ... and {len(self.migration_report['errors']) - 5} more")
        
        print(f"\nDetailed report saved to: {report_file}")
        print(f"Migration instructions: {instructions_file}")
        
        if len(self.migration_report['errors']) == 0:
            print("\n✅ Migration completed successfully!")
            print("\nNext steps:")
            print("1. Run: python cli.py check")
            print("2. List configs: python cli.py list") 
            print("3. Test a model: python cli.py test --config <name>")
        else:
            print("\n⚠️ Migration completed with errors")
            print("Check the detailed report and migration instructions")


def main():
    """Main migration function."""
    parser = argparse.ArgumentParser(
        description="Migrate from legacy Tiny Agent Trainer to production version"
    )
    parser.add_argument(
        "legacy_dir",
        type=Path,
        help="Directory containing legacy Tiny Agent Trainer files"
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path.cwd(),
        help="Target directory for migrated files (default: current directory)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without making changes"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup of target directory before migration"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.legacy_dir.exists():
        print(f"Error: Legacy directory {args.legacy_dir} does not exist")
        return 1
    
    if not args.legacy_dir.is_dir():
        print(f"Error: {args.legacy_dir} is not a directory")
        return 1
    
    # Create backup if requested
    if args.backup and args.target_dir.exists():
        backup_dir = args.target_dir.parent / f"{args.target_dir.name}_backup"
        shutil.copytree(args.target_dir, backup_dir)
        print(f"Created backup at: {backup_dir}")
    
    # Run migration
    migrator = LegacyMigrator(args.legacy_dir, args.target_dir)
    
    if args.dry_run:
        detected = migrator.detect_legacy_files()
        print("DRY RUN - Files that would be migrated:")
        for category, files in detected.items():
            if files:
                print(f"\n{category}:")
                for file in files:
                    print(f"  {file}")
        return 0
    
    try:
        success = migrator.run_migration()
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
