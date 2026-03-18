#!/bin/bash

set -euo pipefail

DEFAULT_ENV_FILE=".env"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEPLOY_STATE_DIR="$PROJECT_ROOT/.deploy"
RELEASE_BRIEF_FILE="$DEPLOY_STATE_DIR/release-brief.md"
RELEASE_BRIEF_USAGE_FILE="$DEPLOY_STATE_DIR/release-brief.last-used.sha256"
DEPLOY_VERSION_FROM=""
DEPLOY_VERSION_TO=""
DEPLOY_VERSION_BUMP_TYPE="unknown"
DEPLOY_RELEASE_BRIEF_MODE="auto" # auto|always|never
DEPLOY_VERSION_OVERRIDE=""
DEPLOY_BUMP_OVERRIDE="patch" # major|minor|patch|none
DEPLOY_NON_INTERACTIVE=0
RELEASE_BRIEF_FRESH=0
RELEASE_BRIEF_CONSUMED=0

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_step() {
  echo -e "${BLUE}$1${NC}"
}

print_success() {
  echo -e "${GREEN}$1${NC}"
}

print_warning() {
  echo -e "${YELLOW}$1${NC}"
}

print_error() {
  echo -e "${RED}$1${NC}"
}

read_release_brief_field() {
  local pattern="$1"
  local value

  [[ -f "$RELEASE_BRIEF_FILE" ]] || return 0
  value="$(
    grep -E -m 1 "^[[:space:]]*(${pattern})[[:space:]]*:" "$RELEASE_BRIEF_FILE" 2>/dev/null \
      | cut -d ':' -f2-
  )"
  value="$(echo "$value" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
  [[ -n "$value" ]] && echo "$value"
}

read_release_brief_summary() {
  [[ -f "$RELEASE_BRIEF_FILE" ]] || return 0

  awk '
    BEGIN { in_summary = 0 }
    /^[[:space:]]*Summary[[:space:]]*:/ {
      in_summary = 1
      sub(/^[^:]*:[[:space:]]*/, "", $0)
      if (length($0) > 0) print $0
      next
    }
    in_summary && /^[[:space:]]*[A-Za-z][A-Za-z0-9 _-]*[[:space:]]*:/ {
      exit
    }
    in_summary {
      print
    }
  ' "$RELEASE_BRIEF_FILE" \
    | sed 's/\r$//' \
    | sed '/^[[:space:]]*$/d' \
    | sed 's/^[[:space:]]*[-*][[:space:]]*/- /; s/^[[:space:]]\+//' \
    | awk 'tolower($0) != "-"'
}

ensure_release_brief_file() {
  if [[ -f "$RELEASE_BRIEF_FILE" ]]; then
    return
  fi
  mkdir -p "$DEPLOY_STATE_DIR"
  cat > "$RELEASE_BRIEF_FILE" <<'EOF'
# Release Brief
Version:
Commit:
Summary:
- 
EOF
}

release_brief_hash() {
  [[ -f "$RELEASE_BRIEF_FILE" ]] || return 0
  hash_file "$RELEASE_BRIEF_FILE"
}

detect_release_brief_freshness() {
  RELEASE_BRIEF_FRESH=0
  [[ -f "$RELEASE_BRIEF_FILE" ]] || return 0

  local current_hash
  local previous_hash=""
  current_hash="$(release_brief_hash)"
  if [[ -f "$RELEASE_BRIEF_USAGE_FILE" ]]; then
    previous_hash="$(cut -d ' ' -f1 < "$RELEASE_BRIEF_USAGE_FILE")"
  fi

  if [[ -z "$previous_hash" || "$previous_hash" != "$current_hash" ]]; then
    RELEASE_BRIEF_FRESH=1
  fi
}

mark_release_brief_used() {
  [[ -f "$RELEASE_BRIEF_FILE" ]] || return 0
  local current_hash
  current_hash="$(release_brief_hash)"
  mkdir -p "$DEPLOY_STATE_DIR"
  printf '%s  %s\n' "$current_hash" "$RELEASE_BRIEF_FILE" > "$RELEASE_BRIEF_USAGE_FILE"
}

should_use_release_brief_for_version() {
  case "$DEPLOY_RELEASE_BRIEF_MODE" in
    always) return 0 ;;
    never) return 1 ;;
    auto)
      [[ "$RELEASE_BRIEF_FRESH" -eq 1 ]]
      return
      ;;
    *)
      return 1
      ;;
  esac
}

should_use_release_brief_for_commit_text() {
  case "$DEPLOY_RELEASE_BRIEF_MODE" in
    always) return 0 ;;
    never) return 1 ;;
    auto) return 1 ;;
    *) return 1 ;;
  esac
}

release_brief_state_label() {
  if [[ ! -f "$RELEASE_BRIEF_FILE" ]]; then
    echo "missing"
    return
  fi

  if [[ "$RELEASE_BRIEF_FRESH" -eq 1 ]]; then
    echo "fresh"
  else
    echo "stale"
  fi
}

print_run_configuration() {
  local env_file="$1"
  local brief_state
  local brief_text_mode="auto-generated"
  local version_override
  local interactive_mode="interactive"
  local env_display
  brief_state="$(release_brief_state_label)"
  if should_use_release_brief_for_commit_text; then
    brief_text_mode="release-brief"
  fi
  if [[ "$DEPLOY_NON_INTERACTIVE" -eq 1 ]]; then
    interactive_mode="non-interactive"
  fi
  version_override="${DEPLOY_VERSION_OVERRIDE:-none}"
  env_display="$env_file"
  if [[ "$env_display" == "$PROJECT_ROOT" ]]; then
    env_display="."
  elif [[ "$env_display" == "$PROJECT_ROOT/"* ]]; then
    env_display="${env_display#$PROJECT_ROOT/}"
  fi

  print_step "Deploy config: env=${env_display}, mode=${interactive_mode}, bump=${DEPLOY_BUMP_OVERRIDE}, version_override=${version_override}, release_brief_mode=${DEPLOY_RELEASE_BRIEF_MODE}, release_brief_state=${brief_state}, commit_text=${brief_text_mode}"
}

hash_file() {
  local file_path="$1"

  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$file_path" | awk '{print $1}'
    return
  fi

  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$file_path" | awk '{print $1}'
    return
  fi

  print_error "Either shasum or sha256sum is required"
  exit 1
}

hash_text() {
  local text="$1"

  if command -v shasum >/dev/null 2>&1; then
    printf '%s' "$text" | shasum -a 256 | awk '{print $1}'
    return
  fi

  if command -v sha256sum >/dev/null 2>&1; then
    printf '%s' "$text" | sha256sum | awk '{print $1}'
    return
  fi

  print_error "Either shasum or sha256sum is required"
  exit 1
}

env_state_file() {
  local env_file="$1"
  local env_abs_path
  local env_id

  env_abs_path="$(cd "$(dirname "$env_file")" && pwd)/$(basename "$env_file")"
  env_id="$(hash_text "$env_abs_path")"
  echo "$DEPLOY_STATE_DIR/env-sync-${env_id}.sha256"
}

prompt_yes_no() {
  local prompt="$1"
  local default="${2:-n}"
  local default_label="no"

  if [[ "$default" == "y" ]]; then
    default_label="yes"
  fi

  if [[ "$DEPLOY_NON_INTERACTIVE" -eq 1 ]]; then
    print_step "Non-interactive: ${prompt} -> ${default_label}"
    [[ "$default" == "y" ]] && return 0 || return 1
  fi

  if [[ "$default" == "y" ]]; then
    prompt="$prompt [Y/n]: "
  else
    prompt="$prompt [y/N]: "
  fi

  while true; do
    read -r -p "$prompt" yn
    case "$yn" in
      [Yy]*) return 0 ;;
      [Nn]*) return 1 ;;
      "")
        [[ "$default" == "y" ]] && return 0 || return 1
        ;;
      *) echo "Please answer yes or no." ;;
    esac
  done
}

print_usage() {
  cat <<'EOF'
Usage: scripts/deploy.sh [env_file] [options]

Options:
  --env-file <path>         Environment file path (default: .env)
  --version <x.y.z>         Explicit version override for this run
  --bump <major|minor|patch|none>
                            Auto bump strategy when --version is not provided (default: patch)
  --non-interactive, -y     Run without prompts (uses defaults + generated commit metadata)
  --use-release-brief       Always use release-brief fields when present
  --ignore-release-brief    Never use release-brief fields
  --help                    Show this help
EOF
}

parse_args() {
  DEPLOY_ENV_FILE="$DEFAULT_ENV_FILE"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --env-file)
        [[ $# -ge 2 ]] || { print_error "Missing value for --env-file"; exit 1; }
        DEPLOY_ENV_FILE="$2"
        shift 2
        ;;
      --version)
        [[ $# -ge 2 ]] || { print_error "Missing value for --version"; exit 1; }
        DEPLOY_VERSION_OVERRIDE="$2"
        shift 2
        ;;
      --bump)
        [[ $# -ge 2 ]] || { print_error "Missing value for --bump"; exit 1; }
        DEPLOY_BUMP_OVERRIDE="$2"
        shift 2
        ;;
      --non-interactive|-y)
        DEPLOY_NON_INTERACTIVE=1
        shift
        ;;
      --use-release-brief)
        DEPLOY_RELEASE_BRIEF_MODE="always"
        shift
        ;;
      --ignore-release-brief|--no-release-brief)
        DEPLOY_RELEASE_BRIEF_MODE="never"
        shift
        ;;
      --help|-h)
        print_usage
        exit 0
        ;;
      -*)
        print_error "Unknown option: $1"
        print_usage
        exit 1
        ;;
      *)
        if [[ "$DEPLOY_ENV_FILE" == "$DEFAULT_ENV_FILE" ]]; then
          DEPLOY_ENV_FILE="$1"
          shift
        else
          print_error "Unexpected positional argument: $1"
          print_usage
          exit 1
        fi
        ;;
    esac
  done
}

check_prerequisites() {
  print_step "Checking prerequisites..."

  command -v git >/dev/null 2>&1 || { print_error "git is required"; exit 1; }
  command -v gh >/dev/null 2>&1 || { print_error "gh is required"; exit 1; }

  if ! gh auth status >/dev/null 2>&1; then
    print_error "GitHub CLI is not authenticated"
    echo "Run: gh auth login"
    exit 1
  fi
}

validate_env_file() {
  local env_file="$1"
  local required=(
    "PGHOST"
    "PGPORT"
    "PGDATABASE"
    "PGUSER"
    "PGPASSWORD"
    "DOCKERHUB_USER"
    "DOCKERHUB_TOKEN"
  )

  print_step "Validating $env_file..."
  for key in "${required[@]}"; do
    if ! grep -q "^${key}=" "$env_file"; then
      print_error "Missing required key: $key"
      exit 1
    fi

    local value
    value=$(grep "^${key}=" "$env_file" | tail -n 1 | cut -d '=' -f2-)
    value=$(echo "$value" | sed 's/^["'\'']\|["'\'']$//g')
    if [[ -z "$value" ]]; then
      print_error "Required key is empty: $key"
      exit 1
    fi
  done
}

increment_version_by_bump() {
  local version="$1"
  local bump="$2"
  local -a parts
  local idx
  local i
  local out

  if [[ ! "$version" =~ ^[0-9]+(\.[0-9]+)+$ ]]; then
    echo "$version"
    return
  fi

  IFS='.' read -r -a parts <<< "$version"
  if [[ "${#parts[@]}" -eq 0 ]]; then
    echo "$version"
    return
  fi

  case "$bump" in
    none)
      echo "$version"
      return
      ;;
    major)
      idx=0
      ;;
    minor)
      if [[ "${#parts[@]}" -lt 2 ]]; then
        parts+=(0)
      fi
      idx=1
      ;;
    patch)
      idx=$((${#parts[@]} - 1))
      ;;
    *)
      print_error "Unsupported bump strategy: $bump (expected major|minor|patch|none)"
      exit 1
      ;;
  esac

  parts[$idx]=$((10#${parts[$idx]} + 1))
  for ((i = idx + 1; i < ${#parts[@]}; i += 1)); do
    parts[$i]=0
  done

  out="${parts[0]}"
  for ((i = 1; i < ${#parts[@]}; i += 1)); do
    out="${out}.${parts[$i]}"
  done
  echo "$out"
}

suggest_next_version() {
  local version="$1"
  local bump="${2:-patch}"
  increment_version_by_bump "$version" "$bump"
}

compare_versions() {
  local current="$1"
  local target="$2"
  local -a current_parts target_parts
  local max_len=0
  local i

  IFS='.' read -r -a current_parts <<< "$current"
  IFS='.' read -r -a target_parts <<< "$target"

  if [[ "${#target_parts[@]}" -gt "$max_len" ]]; then
    max_len="${#target_parts[@]}"
  fi
  if [[ "${#current_parts[@]}" -gt "$max_len" ]]; then
    max_len="${#current_parts[@]}"
  fi

  for ((i = 0; i < max_len; i += 1)); do
    local cur_part="${current_parts[$i]:-0}"
    local new_part="${target_parts[$i]:-0}"
    if ((10#$new_part > 10#$cur_part)); then
      echo "gt"
      return
    fi
    if ((10#$new_part < 10#$cur_part)); then
      echo "lt"
      return
    fi
  done

  echo "eq"
}

infer_bump_type() {
  local current="$1"
  local target="$2"
  local direction
  local -a current_parts target_parts
  local max_len=0
  local first_diff=-1
  local i

  if [[ ! "$current" =~ ^[0-9]+(\.[0-9]+)+$ || ! "$target" =~ ^[0-9]+(\.[0-9]+)+$ ]]; then
    echo "custom"
    return
  fi

  direction="$(compare_versions "$current" "$target")"
  if [[ "$direction" == "eq" ]]; then
    echo "unchanged"
    return
  fi
  if [[ "$direction" == "lt" ]]; then
    echo "downgrade"
    return
  fi

  IFS='.' read -r -a current_parts <<< "$current"
  IFS='.' read -r -a target_parts <<< "$target"

  if [[ "${#target_parts[@]}" -gt "$max_len" ]]; then
    max_len="${#target_parts[@]}"
  fi
  if [[ "${#current_parts[@]}" -gt "$max_len" ]]; then
    max_len="${#current_parts[@]}"
  fi

  for ((i = 0; i < max_len; i += 1)); do
    local cur_part="${current_parts[$i]:-0}"
    local new_part="${target_parts[$i]:-0}"
    if ((10#$new_part != 10#$cur_part)); then
      first_diff="$i"
      break
    fi
  done

  if [[ "$first_diff" -lt 0 ]]; then
    echo "custom"
    return
  fi

  if [[ "$first_diff" -eq 0 ]]; then
    for ((i = 1; i < max_len; i += 1)); do
      if ((10#${target_parts[$i]:-0} != 0)); then
        echo "custom"
        return
      fi
    done
    echo "major"
    return
  fi

  if [[ "$first_diff" -eq 1 ]]; then
    for ((i = 2; i < max_len; i += 1)); do
      if ((10#${target_parts[$i]:-0} != 0)); then
        echo "custom"
        return
      fi
    done
    echo "minor"
    return
  fi

  for ((i = first_diff + 1; i < max_len; i += 1)); do
    if ((10#${target_parts[$i]:-0} != 0)); then
      echo "custom"
      return
    fi
  done
  echo "patch"
}

set_version() {
  local current_version
  local suggested_version
  local patch_version
  local minor_version
  local major_version
  local release_brief_version
  local release_brief_direction=""
  local release_brief_note=""
  local default_choice=""
  local selected_choice=""
  local new_version
  local version_source="auto-bump"
  current_version=$(tr -d '[:space:]' < VERSION 2>/dev/null || true)
  current_version="${current_version:-0.1.0}"
  patch_version="$(suggest_next_version "$current_version" "patch")"
  minor_version="$(suggest_next_version "$current_version" "minor")"
  major_version="$(suggest_next_version "$current_version" "major")"
  suggested_version="$(suggest_next_version "$current_version" "$DEPLOY_BUMP_OVERRIDE")"
  release_brief_version=""

  if should_use_release_brief_for_version; then
    release_brief_version="$(read_release_brief_field "Version" || true)"
    if [[ -n "$release_brief_version" ]]; then
      if [[ "$release_brief_version" =~ ^[0-9]+(\.[0-9]+)+$ ]]; then
        print_step "Version hint loaded from .deploy/release-brief.md: $release_brief_version"
        release_brief_direction="$(compare_versions "$current_version" "$release_brief_version")"
        case "$release_brief_direction" in
          gt)
            release_brief_note="ahead"
            if [[ "$DEPLOY_RELEASE_BRIEF_MODE" == "always" ]]; then
              suggested_version="$release_brief_version"
              version_source="release-brief"
            fi
            ;;
          eq)
            release_brief_note="same as current"
            if [[ "$DEPLOY_RELEASE_BRIEF_MODE" == "always" ]]; then
              suggested_version="$release_brief_version"
              version_source="release-brief"
            fi
            ;;
          lt)
            release_brief_note="downgrade"
            if [[ "$DEPLOY_RELEASE_BRIEF_MODE" == "always" ]]; then
              suggested_version="$release_brief_version"
              version_source="release-brief"
            else
              print_warning "Ignoring release-brief version for auto selection because it would downgrade (${current_version} -> ${release_brief_version})."
            fi
            ;;
        esac
      else
        print_warning "Ignoring invalid Version in .deploy/release-brief.md: $release_brief_version"
        release_brief_version=""
      fi
    fi
  fi

  if [[ -n "$DEPLOY_VERSION_OVERRIDE" ]]; then
    suggested_version="$DEPLOY_VERSION_OVERRIDE"
    version_source="cli-override"
  fi

  if [[ -z "$DEPLOY_VERSION_OVERRIDE" && -t 0 && "$DEPLOY_NON_INTERACTIVE" -eq 0 ]]; then
    case "$version_source" in
      release-brief) default_choice="5" ;;
      auto-bump)
        case "$DEPLOY_BUMP_OVERRIDE" in
          major) default_choice="3" ;;
          minor) default_choice="2" ;;
          none) default_choice="4" ;;
          *) default_choice="1" ;;
        esac
        ;;
      *) default_choice="1" ;;
    esac

    echo ""
    print_step "SemVer version selection:"
    echo "  Current: ${current_version}"
    echo "  1) patch  -> ${patch_version} (backward-compatible fixes)"
    echo "  2) minor  -> ${minor_version} (backward-compatible features)"
    echo "  3) major  -> ${major_version} (breaking changes)"
    echo "  4) keep   -> ${current_version} (no version bump)"
    if [[ -n "$release_brief_version" ]]; then
      echo "  5) release-brief -> ${release_brief_version} (${release_brief_note:-from release brief})"
    else
      echo "  5) release-brief -> unavailable"
    fi
    echo "  6) custom version"
    read -r -p "Select option [${default_choice}]: " selected_choice
    selected_choice="${selected_choice:-$default_choice}"

    case "$selected_choice" in
      1)
        suggested_version="$patch_version"
        version_source="semver-patch"
        ;;
      2)
        suggested_version="$minor_version"
        version_source="semver-minor"
        ;;
      3)
        suggested_version="$major_version"
        version_source="semver-major"
        ;;
      4)
        suggested_version="$current_version"
        version_source="keep-current"
        ;;
      5)
        if [[ -z "$release_brief_version" ]]; then
          print_error "Release-brief version is unavailable."
          exit 1
        fi
        suggested_version="$release_brief_version"
        version_source="release-brief"
        RELEASE_BRIEF_CONSUMED=1
        ;;
      6)
        read -r -p "Enter custom version [${suggested_version}]: " new_version
        suggested_version="${new_version:-$suggested_version}"
        version_source="custom"
        ;;
      *)
        print_error "Invalid version selection: ${selected_choice}"
        exit 1
        ;;
    esac
  fi

  new_version="$suggested_version"
  if [[ "$version_source" == "release-brief" ]]; then
    RELEASE_BRIEF_CONSUMED=1
  fi

  if [[ ! "$new_version" =~ ^[0-9]+(\.[0-9]+)+$ ]]; then
    print_error "Invalid version format: $new_version (expected dot-separated numeric segments)"
    exit 1
  fi

  local bump_type
  bump_type="$(infer_bump_type "$current_version" "$new_version")"
  print_step "Detected version bump type: $bump_type ($current_version -> $new_version) via $version_source"

  case "$bump_type" in
    downgrade)
      print_warning "Selected version is lower than current."
      if ! prompt_yes_no "Continue with downgrade?" "n"; then
        print_error "Version downgrade cancelled."
        exit 1
      fi
      ;;
    major)
      print_warning "Major bump detected."
      if ! prompt_yes_no "Continue with major bump?" "y"; then
        print_error "Major version bump cancelled."
        exit 1
      fi
      ;;
    custom)
      print_warning "Custom multi-segment bump detected."
      if ! prompt_yes_no "Continue with custom bump?" "y"; then
        print_error "Custom version bump cancelled."
        exit 1
      fi
      ;;
  esac

  DEPLOY_VERSION_FROM="$current_version"
  DEPLOY_VERSION_TO="$new_version"
  DEPLOY_VERSION_BUMP_TYPE="$bump_type"

  if [[ "$new_version" != "$current_version" ]]; then
    echo "$new_version" > VERSION
    print_success "Updated VERSION to $new_version"
  else
    print_step "Keeping VERSION at $current_version"
  fi
}

upload_secrets() {
  local env_file="$1"
  local state_file
  local current_hash
  local previous_hash=""

  state_file="$(env_state_file "$env_file")"
  current_hash="$(hash_file "$env_file")"

  if [[ -f "$state_file" ]]; then
    previous_hash="$(cut -d ' ' -f1 < "$state_file")"
  fi

  if [[ -n "$previous_hash" && "$previous_hash" == "$current_hash" ]]; then
    print_step "Environment file unchanged; skipping GitHub secret upload."
    return
  fi

  print_step "Uploading GitHub secrets from $env_file..."
  "$SCRIPT_DIR/setup-secrets.sh" "$env_file"
  mkdir -p "$DEPLOY_STATE_DIR"
  printf '%s  %s\n' "$current_hash" "$env_file" > "$state_file"
  print_success "Recorded env sync fingerprint for $env_file"
}

generate_commit_message() {
  local release_brief_commit
  if should_use_release_brief_for_commit_text; then
    release_brief_commit="$(read_release_brief_field "Commit|Subject" || true)"
    if [[ -n "$release_brief_commit" ]]; then
      RELEASE_BRIEF_CONSUMED=1
      echo "$release_brief_commit"
      return
    fi
  fi

  local staged_files
  staged_files="$(git diff --cached --name-only)"

  if [[ -z "$staged_files" ]]; then
    echo "Deploy demo-fastapi"
    return
  fi

  local file_count=0
  local non_version_count=0
  local has_app=0
  local has_tests=0
  local has_docs=0
  local has_scripts=0
  local has_deploy=0
  local has_db=0
  local has_version=0
  local target_first=""
  local target_second=""
  local path

  while IFS= read -r path; do
    [[ -z "$path" ]] && continue
    ((file_count += 1))
    case "$path" in
      src/*|main.py) has_app=1 ;;
      tests/*) has_tests=1 ;;
      docs/*|README.md) has_docs=1 ;;
      scripts/*) has_scripts=1 ;;
      .github/*|k8s-*|Dockerfile|Dockerfile.runner|sonar-project.properties) has_deploy=1 ;;
      sql/*) has_db=1 ;;
      VERSION) has_version=1 ;;
    esac

    if [[ "$path" != "VERSION" ]]; then
      ((non_version_count += 1))
      local base_name
      base_name="$(basename "$path")"

      if [[ -z "$target_first" ]]; then
        target_first="$base_name"
      elif [[ "$base_name" != "$target_first" && -z "$target_second" ]]; then
        target_second="$base_name"
      fi
    fi
  done <<< "$staged_files"

  local scope_text=""
  [[ "$has_app" -eq 1 ]] && scope_text="${scope_text:+$scope_text, }workspace"
  [[ "$has_db" -eq 1 ]] && scope_text="${scope_text:+$scope_text, }db"
  [[ "$has_scripts" -eq 1 ]] && scope_text="${scope_text:+$scope_text, }scripts"
  [[ "$has_deploy" -eq 1 ]] && scope_text="${scope_text:+$scope_text, }deploy"
  [[ "$has_tests" -eq 1 ]] && scope_text="${scope_text:+$scope_text, }tests"
  [[ "$has_docs" -eq 1 ]] && scope_text="${scope_text:+$scope_text, }docs"
  if [[ -z "$scope_text" ]]; then
    if [[ "$has_version" -eq 1 ]]; then
      scope_text="version"
    else
      scope_text="updates"
    fi
  fi

  local current_version
  current_version="$(tr -d '[:space:]' < VERSION 2>/dev/null || true)"

  if [[ "$has_version" -eq 1 && "$file_count" -eq 1 && -n "$current_version" ]]; then
    echo "chore: bump VERSION to ${current_version}"
    return
  fi

  local target_text="staged changes"
  local captured_targets=0
  [[ -n "$target_first" ]] && ((captured_targets += 1))
  [[ -n "$target_second" ]] && ((captured_targets += 1))
  local target_suffix_count=$((non_version_count - captured_targets))
  if [[ "$captured_targets" -eq 1 ]]; then
    target_text="$target_first"
  elif [[ "$captured_targets" -ge 2 ]]; then
    target_text="${target_first} and ${target_second}"
  fi
  if [[ "$target_suffix_count" -gt 0 ]]; then
    target_text="${target_text} +${target_suffix_count} files"
  fi

  if [[ -n "$current_version" ]]; then
    echo "deploy(${scope_text}): ${target_text} (v${current_version})"
  else
    echo "deploy(${scope_text}): ${target_text}"
  fi
}

generate_staged_summary() {
  local staged_status
  local line_count=0
  local summary=""
  local status
  local path
  local rest

  staged_status="$(git diff --cached --name-status)"
  while IFS=$'\t' read -r status path rest; do
    [[ -z "$status" ]] && continue
    case "$status" in
      A) summary="${summary}- Added ${path}\n" ;;
      M) summary="${summary}- Updated ${path}\n" ;;
      D) summary="${summary}- Removed ${path}\n" ;;
      R*) summary="${summary}- Renamed ${path} -> ${rest}\n" ;;
      *) summary="${summary}- Changed ${path}\n" ;;
    esac
    ((line_count += 1))
    if [[ "$line_count" -ge 8 ]]; then
      break
    fi
  done <<< "$staged_status"

  if [[ "$line_count" -eq 0 ]]; then
    echo "- No staged file summary available."
    return
  fi

  printf '%b' "$summary"
}

generate_commit_body() {
  local summary_text
  local current_version
  local staged_count
  local staged_paths
  local metadata_block
  local body=""

  if should_use_release_brief_for_commit_text; then
    summary_text="$(read_release_brief_summary || true)"
    if [[ -n "$summary_text" ]]; then
      RELEASE_BRIEF_CONSUMED=1
    fi
  fi
  if [[ -z "$summary_text" ]]; then
    summary_text="$(generate_staged_summary)"
  fi
  current_version="$(tr -d '[:space:]' < VERSION 2>/dev/null || true)"
  staged_count="$(git diff --cached --name-only | sed '/^[[:space:]]*$/d' | wc -l | tr -d ' ')"
  staged_paths="$(git diff --cached --name-only | sed '/^[[:space:]]*$/d' | head -n 8 | sed 's/^/- /')"

  if [[ -n "$summary_text" ]]; then
    body="AI release summary:
$summary_text"
  fi

  metadata_block="Release metadata:
- version: ${current_version:-n/a}
- staged files: ${staged_count:-0}"
  if [[ -n "$DEPLOY_VERSION_BUMP_TYPE" && "$DEPLOY_VERSION_BUMP_TYPE" != "unknown" ]]; then
    metadata_block="${metadata_block}
- version bump: ${DEPLOY_VERSION_BUMP_TYPE} (${DEPLOY_VERSION_FROM:-$current_version} -> ${DEPLOY_VERSION_TO:-$current_version})"
  fi
  if [[ -n "$staged_paths" ]]; then
    metadata_block="${metadata_block}
- paths:
${staged_paths}"
  fi

  if [[ -n "$body" ]]; then
    body="${body}

${metadata_block}"
  else
    body="${metadata_block}"
  fi

  printf '%s' "$body"
}

write_deploy_notes() {
  local commit_subject="$1"
  local commit_body="${2:-}"
  local notes_file="$DEPLOY_STATE_DIR/deploy-notes-latest.md"
  local generated_at_utc
  local current_version
  local current_branch
  local base_sha
  local shortstat

  mkdir -p "$DEPLOY_STATE_DIR"
  generated_at_utc="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  current_version="$(tr -d '[:space:]' < VERSION 2>/dev/null || true)"
  current_branch="$(git branch --show-current)"
  base_sha="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  shortstat="$(git diff --cached --shortstat | sed 's/^ *//')"
  shortstat="${shortstat:-0 files changed}"

  {
    echo "# Deploy Notes"
    echo ""
    echo "- Generated (UTC): ${generated_at_utc}"
    echo "- Branch: ${current_branch}"
    echo "- Base SHA: ${base_sha}"
    if [[ -n "$current_version" ]]; then
      echo "- Version: ${current_version}"
    fi
    if [[ -n "$DEPLOY_VERSION_BUMP_TYPE" && "$DEPLOY_VERSION_BUMP_TYPE" != "unknown" ]]; then
      echo "- Version bump: ${DEPLOY_VERSION_BUMP_TYPE} (${DEPLOY_VERSION_FROM:-$current_version} -> ${DEPLOY_VERSION_TO:-$current_version})"
    fi
    echo "- Proposed commit: ${commit_subject}"
    echo "- Diff summary: ${shortstat}"
    echo ""
    echo "## Staged files"
    while IFS= read -r line; do
      [[ -z "$line" ]] && continue
      echo "- \`${line}\`"
    done <<< "$(git diff --cached --name-status)"
    if [[ -f "$RELEASE_BRIEF_FILE" ]]; then
      echo ""
      echo "## Release Brief"
      echo "- Source: \`${RELEASE_BRIEF_FILE#$PROJECT_ROOT/}\`"
      echo ""
      sed 's/^/> /' "$RELEASE_BRIEF_FILE"
    fi
    if [[ -n "$commit_body" ]]; then
      echo ""
      echo "## Proposed Commit Body"
      echo ""
      printf '%s\n' "$commit_body" | sed 's/^/> /'
    fi
  } > "$notes_file"

  echo "$notes_file"
}

append_deploy_history() {
  local commit_subject="$1"
  local history_file="$DEPLOY_STATE_DIR/deploy-history.md"
  local generated_at_utc
  local current_version
  local current_branch
  local head_sha
  local shortstat

  mkdir -p "$DEPLOY_STATE_DIR"
  generated_at_utc="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  current_version="$(tr -d '[:space:]' < VERSION 2>/dev/null || true)"
  current_branch="$(git branch --show-current)"
  head_sha="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  shortstat="$(git show --shortstat --format='' HEAD | sed 's/^ *//')"
  shortstat="${shortstat:-0 files changed}"

  {
    echo "## ${generated_at_utc} | ${commit_subject}"
    echo ""
    echo "- Branch: ${current_branch}"
    echo "- Commit: ${head_sha}"
    if [[ -n "$current_version" ]]; then
      echo "- Version: ${current_version}"
    fi
    if [[ -n "$DEPLOY_VERSION_BUMP_TYPE" && "$DEPLOY_VERSION_BUMP_TYPE" != "unknown" ]]; then
      echo "- Version bump: ${DEPLOY_VERSION_BUMP_TYPE} (${DEPLOY_VERSION_FROM:-$current_version} -> ${DEPLOY_VERSION_TO:-$current_version})"
    fi
    echo "- Summary: ${shortstat}"
    echo "- Files:"
    while IFS= read -r line; do
      [[ -z "$line" ]] && continue
      echo "  - ${line}"
    done <<< "$(git show --name-status --format='' HEAD)"
    if [[ -f "$RELEASE_BRIEF_FILE" ]]; then
      local brief_commit
      brief_commit="$(read_release_brief_field "Commit|Subject" || true)"
      [[ -n "$brief_commit" ]] && echo "- Release brief commit: ${brief_commit}"
    fi
    echo ""
  } >> "$history_file"

  echo "$history_file"
}

commit_and_push() {
  local commit_message
  local auto_commit_message
  local auto_commit_body
  local deploy_notes_file
  local deploy_history_file
  local unstaged_tracked
  local untracked_files

  git status --short

  unstaged_tracked="$(git diff --name-only)"
  untracked_files="$(git ls-files --others --exclude-standard)"

  if [[ -n "$untracked_files" ]]; then
    print_error "Refusing to deploy with untracked files present."
    echo "$untracked_files"
    echo ""
    echo "Stage or ignore these files intentionally before deploying."
    exit 1
  fi

  if [[ -n "$unstaged_tracked" ]]; then
    if [[ "$unstaged_tracked" == "VERSION" ]]; then
      git add VERSION
    else
      print_error "Refusing to deploy with unstaged tracked changes."
      echo "$unstaged_tracked"
      echo ""
      echo "Stage exactly what you want deployed, then rerun the script."
      exit 1
    fi
  fi

  if git diff --cached --quiet; then
    print_warning "No staged changes to commit."
    if prompt_yes_no "Trigger deploy workflow manually instead?" "y"; then
      gh workflow run deploy-self-hosted.yaml
      print_success "Workflow dispatch requested."
      return
    fi
    print_warning "Skipping push."
    return
  fi

  auto_commit_message="$(generate_commit_message)"
  auto_commit_body="$(generate_commit_body)"
  local release_brief_commit
  release_brief_commit=""
  if should_use_release_brief_for_commit_text; then
    release_brief_commit="$(read_release_brief_field "Commit|Subject" || true)"
  fi
  if [[ -n "$release_brief_commit" && "$auto_commit_message" == "$release_brief_commit" ]]; then
    print_step "Commit subject loaded from .deploy/release-brief.md"
  fi
  deploy_notes_file="$(write_deploy_notes "$auto_commit_message" "$auto_commit_body")"
  print_step "Generated deploy notes: ${deploy_notes_file#$PROJECT_ROOT/}"
  if [[ -n "$auto_commit_body" ]]; then
    echo ""
    print_step "AI commit body preview:"
    printf '%s\n' "$auto_commit_body"
  fi
  if [[ "$DEPLOY_NON_INTERACTIVE" -eq 1 ]]; then
    commit_message="$auto_commit_message"
    print_step "Non-interactive: using generated commit message."
  else
    echo ""
    read -r -p "Commit message [$auto_commit_message]: " commit_message
    commit_message="${commit_message:-$auto_commit_message}"
  fi

  if [[ -n "$auto_commit_body" ]]; then
    git commit -m "$commit_message" -m "$auto_commit_body"
  else
    git commit -m "$commit_message"
  fi
  git push origin "$(git branch --show-current)"
  if [[ "$RELEASE_BRIEF_CONSUMED" -eq 1 ]]; then
    mark_release_brief_used
  fi
  deploy_history_file="$(append_deploy_history "$commit_message")"
  print_step "Updated deploy history: ${deploy_history_file#$PROJECT_ROOT/}"
}

main() {
  cd "$PROJECT_ROOT"

  parse_args "$@"
  local env_file="$DEPLOY_ENV_FILE"

  if [[ ! -f "$env_file" ]]; then
    print_error "Environment file not found: $env_file"
    exit 1
  fi

  check_prerequisites
  ensure_release_brief_file
  detect_release_brief_freshness
  print_run_configuration "$env_file"
  if [[ "$DEPLOY_RELEASE_BRIEF_MODE" == "auto" && "$RELEASE_BRIEF_FRESH" -eq 0 ]]; then
    print_step "Release brief unchanged since last use; using auto-generated version/message/body."
  elif [[ "$DEPLOY_RELEASE_BRIEF_MODE" == "auto" && "$RELEASE_BRIEF_FRESH" -eq 1 ]]; then
    print_step "Release brief is fresh; using it as a version hint only. Commit message/body will be auto-generated unless --use-release-brief is passed."
  fi
  validate_env_file "$env_file"
  set_version
  upload_secrets "$env_file"

  if prompt_yes_no "Commit and push changes to trigger deployment?" "y"; then
    commit_and_push
    print_success "Deployment push complete. Monitor GitHub Actions for progress."
  else
    print_warning "Secrets uploaded. Commit/push skipped."
  fi
}

main "$@"
